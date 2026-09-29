"""
llm_as_a_judge_gpt4o_rag_metrics_pharmaqa.py

Valutazione LLM-as-a-Judge delle metriche RAG complete su PharmaQA IT
tramite GPT-4o-mini (o altro modello OpenAI-compatible).

Metriche calcolate:
    - Faithfulness: estrazione statement + verifica contro il contesto
    - Context Relevance: frasi rilevanti estratte / frasi totali
    - Answer Relevance: domande sintetiche + cosine similarity via embedding

Pipeline AR (stile RAGAS):
    risposta generata -> N domande sintetiche in italiano (LLM)
    embedding(domanda originale) vs embedding(domande sintetiche)
    AR = media delle cosine similarity

Input atteso: lista JSON con campi example_id, dataset, model_id,
    question, generated, context. Campi opzionali: answers, is_correct_string.

Output:
    <output-dir>/rag_metrics_details.json
    <output-dir>/rag_metrics_summary_by_model_dataset.json

Dipendenze: openai (async), sentence-transformers, numpy, dotenv.

Uso:
    python llm_as_a_judge_gpt4o_rag_metrics_pharmaqa.py \\
      --input-file data/llm_as_a_judge/pharmaqa/ragass_samples_pharmaqa.json \\
      --output-dir data/llm_as_a_judge/pharmaqa_rag_metrics \\
      --judge-model gpt-4o-mini --concurrency 16 --datasets pharmaqa_it
"""

import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Caricamento variabili d'ambiente per l'API
load_dotenv(dotenv_path=".env", override=True)

API_URL = os.getenv("LLM_API_URL", "").strip()
API_TOKEN = os.getenv("LLM_API_TOKEN", "").strip()

# ─── Configurazione globale ──────────────────────────────────────────────────
JUDGE_MODEL = "gpt-4o-mini"          # Modello judge per le chiamate LLM
CONCURRENCY = 4                       # Richieste concorrenti verso l'API
N_QUESTIONS = 5                       # Domande sintetiche per Answer Relevance
CTX_MAX_CHARS = 25000                 # Troncamento massimo del contesto
EMBEDDING_MODEL = "BAAI/bge-m3"       # Modello embedding locale

MAX_TOKENS_FAITH = 300                # Max token per le chiamate Faithfulness
MAX_TOKENS_AR_QGEN = 80              # Max token per generazione domande sintetiche
MAX_TOKENS_CR = 400                   # Max token per Context Relevance

DEFAULT_INPUT = Path("data/llm_as_a_judge/pharmaqa/ragass_samples_pharmaqa.json")
DEFAULT_OUTPUT_DIR = Path("data/llm_as_a_judge/pharmaqa_rag_metrics")
DATASETS = {"pharmaqa_it"}

# Variabili globali inizializzate in main()
sem = None                            # Semaforo asyncio per il rate limiting
embedder = None                       # Modello SentenceTransformer
embed_lock = None                     # Lock per serializzare gli embedding
question_embedding_cache = {}         # Cache embedding delle domande originali


def _normalize_openai_base_url(url: str) -> str:
    """Normalizza l'URL base dell'API aggiungendo /v1 se necessario."""
    if not url:
        raise RuntimeError("LLM_API_URL non impostata nel file .env o nell'ambiente.")
    url = url.rstrip("/")
    for suffix in ["/chat/completions", "/v1/chat/completions"]:
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def _clean_api_key(token: str) -> str:
    """Rimuove il prefisso 'Bearer ' se presente nel token API."""
    if not token:
        raise RuntimeError("LLM_API_TOKEN non impostata nel file .env o nell'ambiente.")
    token = token.strip()
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()
    return token


def make_client() -> AsyncOpenAI:
    """Crea un client AsyncOpenAI con URL e token dal file .env."""
    return AsyncOpenAI(
        api_key=_clean_api_key(API_TOKEN),
        base_url=_normalize_openai_base_url(API_URL),
    )


async def llm_call(
    client: AsyncOpenAI,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    """Chiamata singola al modello judge. Rimuove eventuali tag </think>."""
    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = (resp.choices[0].message.content or "").strip()
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


def count_sentences(text: str) -> int:
    """Conta le frasi nel testo (split su .!?) — minimo 1."""
    sentences = re.split(r"[.!?]+", text or "")
    return max(1, len([s for s in sentences if s.strip()]))


# ─── Faithfulness ─────────────────────────────────────────────────────────────
# Prompt di sistema per estrazione e verifica degli statement fattuali
_SYS_FAITH_EXTRACT = (
    "Estrai affermazioni fattuali atomiche da risposte in italiano. "
    "Elenca ogni affermazione su una nuova riga iniziando con 'statement: '."
)

_SYS_FAITH_VERIFY = (
    "Verifica se le affermazioni sono supportate dal contesto. "
    "Per ogni affermazione dai una breve spiegazione, poi un verdetto. "
    "Termina con verdetti in questo formato esatto, uno per riga:\n"
    "verdict: Yes\nverdict: No"
)


def parse_statements(text: str) -> list[str]:
    """Estrae le affermazioni dall'output LLM (formato 'statement:' o elenco numerato)."""
    statements = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.lower().startswith("statement:"):
            s = line[len("statement:"):].strip()
            if s:
                statements.append(s)

    if not statements:
        for line in text.strip().split("\n"):
            line = line.strip()
            m = re.match(r"^[\d]+[.)]\s+(.+)$", line)
            if m:
                statements.append(m.group(1).strip())
            elif line.startswith("- ") and len(line) > 2:
                statements.append(line[2:].strip())

    return statements


def parse_verdicts(text: str, n: int) -> list[bool]:
    """Estrae n verdetti Yes/No dall'output LLM. Fallback: conta occorrenze yes/no."""
    verdicts = []
    for line in text.strip().split("\n"):
        line = line.strip().lower()
        if line.startswith("verdict:"):
            val = line[len("verdict:"):].strip()
            verdicts.append(val.startswith("yes"))

    if len(verdicts) != n:
        yeses = len(re.findall(r"\byes\b", text.lower()))
        nos = len(re.findall(r"\bno\b", text.lower()))
        if yeses + nos:
            verdicts = [True] * min(yeses, n) + [False] * max(0, n - yeses)
        else:
            verdicts = [False] * n

    return (verdicts + [False] * n)[:n]


async def compute_faithfulness(client: AsyncOpenAI, question: str, generated: str, context: str) -> dict:
    """Calcola faithfulness: estrae statement dalla risposta e li verifica contro il contesto."""
    generated = str(generated or "")
    context = str(context or "")

    if not generated.strip():
        return {
            "faithfulness": 1.0,
            "n_statements": 0,
            "n_verified": 0,
            "_faith_statements_raw": "",
            "_faith_verdicts_raw": "",
        }

    gen_trunc = generated[:800]
    user_extract = (
        "Data una domanda e una risposta, crea una o più affermazioni fattuali atomiche "
        "a partire dalla risposta. Ignora testo puramente stilistico o esitazioni.\n"
        f"Domanda: {question}\n"
        f"Risposta: {gen_trunc}\n\n"
        "Elenca ogni affermazione iniziando con 'statement: '."
    )
    raw_stmts = await llm_call(client, _SYS_FAITH_EXTRACT, user_extract, MAX_TOKENS_FAITH)
    statements = parse_statements(raw_stmts)

    if not statements:
        return {
            "faithfulness": 1.0,
            "n_statements": 0,
            "n_verified": 0,
            "_faith_statements_raw": raw_stmts,
            "_faith_verdicts_raw": "",
        }

    stmts_str = "\n".join(f"statement: {s}" for s in statements)
    context_trunc = context[:CTX_MAX_CHARS]
    user_verify = (
        "Considera il contesto e determina se ogni affermazione è supportata. "
        "Una affermazione è supportata solo se il contesto contiene informazioni sufficienti per inferirla. "
        "Usa il significato, non la corrispondenza esatta delle parole.\n\n"
        f"Contesto:\n{context_trunc}\n\n"
        f"Affermazioni:\n{stmts_str}\n\n"
        "Termina con i verdetti finali, uno per affermazione, nello stesso ordine:\n"
        "verdict: Yes\nverdict: No"
    )
    raw_verds = await llm_call(client, _SYS_FAITH_VERIFY, user_verify, MAX_TOKENS_FAITH)
    verdicts = parse_verdicts(raw_verds, len(statements))
    n_verified = sum(verdicts)

    return {
        "faithfulness": round(n_verified / len(statements), 6),
        "n_statements": len(statements),
        "n_verified": n_verified,
        "_faith_statements_raw": raw_stmts,
        "_faith_verdicts_raw": raw_verds,
    }


# ─── Context Relevance ────────────────────────────────────────────────────────
# Prompt di sistema per estrazione delle frasi rilevanti dal contesto
_SYS_CTXREL = (
    "Estrai dal contesto le frasi rilevanti per rispondere alla domanda. "
    "Output SOLO con frasi copiate testualmente, oppure 'Insufficient Information' se nessuna frase è rilevante."
)


async def compute_context_relevance(client: AsyncOpenAI, question: str, context: str) -> dict:
    """Calcola context relevance: frasi rilevanti estratte / frasi totali nel contesto."""
    context = str(context or "")
    if not context.strip():
        return {"context_relevance": 0.0, "n_extracted": 0, "n_total_sentences": 0, "_cr_raw": ""}

    context_trunc = context[:CTX_MAX_CHARS]
    user = (
        "Dal contesto seguente, copia solo le frasi utili per rispondere alla domanda. "
        "Se nessuna frase è utile, scrivi esattamente 'Insufficient Information'. "
        "Non riscrivere le frasi.\n\n"
        f"Domanda: {question}\n"
        f"Contesto:\n{context_trunc}\n\n"
        "Output:"
    )
    raw = await llm_call(client, _SYS_CTXREL, user, MAX_TOKENS_CR)
    n_extracted = 0 if "insufficient information" in raw.lower() else count_sentences(raw)
    n_total = count_sentences(context_trunc)
    score = min(1.0, n_extracted / n_total) if n_total > 0 else 0.0

    return {
        "context_relevance": round(score, 6),
        "n_extracted": n_extracted,
        "n_total_sentences": n_total,
        "_cr_raw": raw,
    }


# ─── Answer Relevance ─────────────────────────────────────────────────────────
# Genera N domande sintetiche, poi calcola cosine similarity con embedding locali
_SYS_QGEN = (
    "Genera una domanda in italiano a cui la risposta data potrebbe rispondere. "
    "Output SOLO con la domanda, nient’altro."
)


async def generate_one_question(client: AsyncOpenAI, generated: str) -> str:
    """Genera una domanda sintetica in italiano a partire dalla risposta generata."""
    gen_trunc = str(generated or "")[:800]
    user = (
        "Genera una domanda plausibile in italiano la cui risposta sarebbe il testo seguente. "
        "Mantieni la domanda specifica. Output solo con la domanda.\n\n"
        f"Risposta:\n{gen_trunc}"
    )
    return await llm_call(client, _SYS_QGEN, user, MAX_TOKENS_AR_QGEN, temperature=0.7)


def _clean_synthetic_question(q: str) -> str:
    """Pulisce una domanda sintetica: rimuove numerazione, apici, spazi."""
    q = (q or "").strip()
    q = re.sub(r"^[\d]+[.)]\s*", "", q).strip()
    q = q.strip('"').strip("'").strip()
    return q


async def embed_texts(texts: list[str]) -> np.ndarray:
    """Calcola embedding di una lista di testi (serializzato tramite lock)."""
    global embedder, embed_lock
    async with embed_lock:
        return await asyncio.to_thread(
            embedder.encode,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


async def get_question_embedding(question: str) -> np.ndarray:
    """Restituisce l'embedding della domanda originale (con cache)."""
    key = question.strip()
    cached = question_embedding_cache.get(key)
    if cached is not None:
        return cached
    emb = (await embed_texts([key]))[0]
    question_embedding_cache[key] = emb
    return emb


async def compute_answer_relevance(client: AsyncOpenAI, question: str, generated: str) -> dict:
    """Calcola AR: genera N domande sintetiche, poi media cosine similarity con la domanda originale."""
    generated = str(generated or "")
    if not generated.strip():
        return {
            "answer_relevance": 0.0,
            "n_synthetic_questions": 0,
            "synthetic_questions": [],
            "synthetic_question_scores": [],
            "_ar_qgen_raw": [],
        }

    q_tasks = [generate_one_question(client, generated) for _ in range(N_QUESTIONS)]
    raw_questions = await asyncio.gather(*q_tasks)

    synthetic_questions = []
    for q in raw_questions:
        q = _clean_synthetic_question(q)
        if q and len(q) >= 5:
            synthetic_questions.append(q)

    synthetic_questions = synthetic_questions[:N_QUESTIONS]
    if not synthetic_questions:
        return {
            "answer_relevance": 0.0,
            "n_synthetic_questions": 0,
            "synthetic_questions": [],
            "synthetic_question_scores": [],
            "_ar_qgen_raw": raw_questions,
        }

    original_emb = await get_question_embedding(str(question or ""))
    synth_embs = await embed_texts(synthetic_questions)
    scores = [float(np.dot(original_emb, emb)) for emb in synth_embs]

    return {
        "answer_relevance": round(float(np.mean(scores)), 6),
        "n_synthetic_questions": len(synthetic_questions),
        "synthetic_questions": synthetic_questions,
        "synthetic_question_scores": [round(s, 6) for s in scores],
        "_ar_qgen_raw": raw_questions,
    }


def record_key(rec: dict) -> tuple:
    """Chiave univoca di un record: (dataset, model_id, example_id)."""
    return (rec.get("dataset"), rec.get("model_id"), rec.get("example_id"))


def is_valid_result(rec: dict) -> bool:
    """True se il record ha tutte le metriche calcolate e nessun errore."""
    if rec.get("error"):
        return False
    required = ["faithfulness", "context_relevance", "answer_relevance"]
    return all(rec.get(k) is not None for k in required)


async def evaluate_record(client: AsyncOpenAI, rec: dict) -> dict:
    """Valuta un singolo record: calcola faithfulness, CR e AR, restituisce il record arricchito."""
    async with sem:
        try:
            question = rec["question"]
            generated = rec.get("generated", "")
            context = rec.get("context", "")

            faith = await compute_faithfulness(client, question, generated, context)
            cr = await compute_context_relevance(client, question, context)
            ar = await compute_answer_relevance(client, question, generated)
            error = None
        except Exception as e:
            faith = {"faithfulness": None, "n_statements": 0, "n_verified": 0, "_faith_statements_raw": "", "_faith_verdicts_raw": ""}
            cr = {"context_relevance": None, "n_extracted": 0, "n_total_sentences": 0, "_cr_raw": ""}
            ar = {"answer_relevance": None, "n_synthetic_questions": 0, "synthetic_questions": [], "synthetic_question_scores": [], "_ar_qgen_raw": []}
            error = f"{type(e).__name__}: {str(e)[:300]}"

        return {
            "example_id": rec.get("example_id"),
            "dataset": rec.get("dataset"),
            "model_id": rec.get("model_id"),
            "question": rec.get("question"),
            "generated": rec.get("generated"),
            "answers": rec.get("answers"),
            "context": rec.get("context"),
            "is_correct_string": rec.get("is_correct_string"),
            **faith,
            **cr,
            **ar,
            "error": error,
        }


def load_existing_details(details_path: Path, target_records: list) -> tuple[list, list]:
    """Carica risultati precedenti per il resume. Ritorna (gia_fatti, da_fare)."""
    if not details_path.exists():
        return [], target_records

    target_keys = {record_key(r) for r in target_records}

    try:
        previous = json.loads(details_path.read_text(encoding="utf-8"))
    except Exception:
        return [], target_records

    done = {}
    for r in previous:
        key = record_key(r)
        if key not in target_keys:
            continue
        if not is_valid_result(r):
            continue
        done[key] = r

    existing = [done[record_key(r)] for r in target_records if record_key(r) in done]
    pending = [r for r in target_records if record_key(r) not in done]

    print(f"Resume: {len(existing)} record già validi, {len(pending)} da valutare.")
    return existing, pending


def build_summary(details: list) -> list:
    """Aggrega i dettagli per (model_id, dataset) e calcola le medie delle metriche."""
    grouped = defaultdict(lambda: {
        "n": 0,
        "n_errors": 0,
        "faithfulness_sum": 0.0,
        "context_relevance_sum": 0.0,
        "answer_relevance_sum": 0.0,
        "n_statements_sum": 0,
        "n_verified_sum": 0,
        "n_synthetic_questions_sum": 0,
    })

    for r in details:
        key = (r.get("model_id"), r.get("dataset"))
        g = grouped[key]

        if not is_valid_result(r):
            g["n_errors"] += 1
            continue

        g["n"] += 1
        g["faithfulness_sum"] += float(r["faithfulness"])
        g["context_relevance_sum"] += float(r["context_relevance"])
        g["answer_relevance_sum"] += float(r["answer_relevance"])
        g["n_statements_sum"] += int(r.get("n_statements", 0))
        g["n_verified_sum"] += int(r.get("n_verified", 0))
        g["n_synthetic_questions_sum"] += int(r.get("n_synthetic_questions", 0))

    summary = []
    for (model_id, dataset), s in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        n = s["n"]
        summary.append({
            "model_id": model_id,
            "dataset": dataset,
            "faithfulness": round(s["faithfulness_sum"] / n, 6) if n else None,
            "context_relevance": round(s["context_relevance_sum"] / n, 6) if n else None,
            "answer_relevance": round(s["answer_relevance_sum"] / n, 6) if n else None,
            "n": n,
            "n_errors": s["n_errors"],
            "n_statements": s["n_statements_sum"],
            "n_verified": s["n_verified_sum"],
            "n_synthetic_questions": s["n_synthetic_questions_sum"],
        })

    return summary


async def run(records: list, details_path: Path, existing_details: list) -> list:
    """Esegue la valutazione asincrona di tutti i record con checkpoint periodico."""
    client = make_client()

    print(f"Judge:          {JUDGE_MODEL}")
    print(f"Embedding:      {EMBEDDING_MODEL}")
    print(f"Record:         {len(records)} nuovi")
    print(f"Concurrency:    {CONCURRENCY}")
    print(f"AR n:           {N_QUESTIONS}")
    print(f"LLM calls:      circa {len(records) * (2 + 1 + N_QUESTIONS)} nuovi")
    print(f"API base:       {_normalize_openai_base_url(API_URL)}")
    print(f"CTX chars:      {CTX_MAX_CHARS}")

    tasks = [evaluate_record(client, rec) for rec in records]
    details = list(existing_details)
    start_done = len(details)
    total = start_done + len(records)

    t0 = time.time()

    async for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks)):
        details.append(await coro)

        if (len(details) - start_done) % 100 == 0:
            details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Checkpoint: {len(details)}/{total}")

    details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"Completato in {elapsed/60:.1f} min.")
    return details


def main():
    """Punto di ingresso: parsing args, caricamento modelli, valutazione, salvataggio."""
    global JUDGE_MODEL, CONCURRENCY, N_QUESTIONS, CTX_MAX_CHARS, EMBEDDING_MODEL
    global sem, embedder, embed_lock

    # Parsing degli argomenti da riga di comando
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--datasets", type=str, nargs="+", default=sorted(DATASETS))
    parser.add_argument("--n-questions", type=int, default=N_QUESTIONS)
    parser.add_argument("--ctx-max-chars", type=int, default=CTX_MAX_CHARS)
    parser.add_argument("--embedding-model", type=str, default=EMBEDDING_MODEL)
    args = parser.parse_args()

    # Aggiornamento delle variabili globali dai parametri CLI
    JUDGE_MODEL = args.judge_model
    CONCURRENCY = args.concurrency
    N_QUESTIONS = args.n_questions
    CTX_MAX_CHARS = args.ctx_max_chars
    EMBEDDING_MODEL = args.embedding_model
    sem = asyncio.Semaphore(CONCURRENCY)
    embed_lock = asyncio.Lock()

    # Inizializzazione del modello di embedding locale
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers non installato. Installa con: pip install sentence-transformers")

    print(f"Caricamento embedding model locale: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Configurazione percorsi di input/output
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    details_path = output_dir / "rag_metrics_details.json"
    summary_path = output_dir / "rag_metrics_summary_by_model_dataset.json"

    # Caricamento e filtraggio dei record di input
    records = json.loads(input_path.read_text(encoding="utf-8"))

    wanted_datasets = set(args.datasets) if args.datasets else set(r.get("dataset") for r in records)
    records = [r for r in records if r.get("dataset") in wanted_datasets]

    if args.models:
        records = [r for r in records if r.get("model_id") in set(args.models)]

    if args.n_samples:
        records = records[:args.n_samples]

    print(f"Input:   {input_path}")
    print(f"Output:  {output_dir}")
    print(f"Dataset: {sorted(wanted_datasets)}")
    print(f"Totale record target: {len(records)}")

    # Resume: separa record gia' completati da quelli da valutare
    existing, pending = load_existing_details(details_path, records)
    details = asyncio.run(run(pending, details_path, existing))

    # Calcolo e salvataggio del summary aggregato per modello/dataset
    summary = build_summary(details)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Salvato dettaglio: {details_path}")
    print(f"Salvato summary:   {summary_path}")


if __name__ == "__main__":
    main()
