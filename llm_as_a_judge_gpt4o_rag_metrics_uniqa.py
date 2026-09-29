"""
llm_as_a_judge_gpt4o_rag_metrics_uniqa.py

Valutazione LLM-as-a-Judge delle metriche RAG complete sul dataset italiano
UniQA (uniqa_it):
    - Faithfulness (fedelta' al contesto)
    - Context Relevance (rilevanza del contesto)
    - Answer Relevance (rilevanza della risposta, stile RAGAS)

I prompt di valutazione sono in italiano. La pipeline di Answer Relevance
segue l'approccio RAGAS:
    risposta generata -> n domande sintetiche in italiano generate dall'LLM
    domanda originale + domande sintetiche -> embedding locali
    answer_relevance = media della similarita' coseno(domanda_originale, domande_sintetiche)

Dataset supportati:
    - uniqa_it

Dipendenze principali:
    - openai (AsyncOpenAI per chiamate asincrone al giudice LLM)
    - python-dotenv (caricamento variabili da .env)
    - asyncio (orchestrazione concorrente)
    - tqdm (barra di avanzamento asincrona)
    - numpy (calcoli vettoriali per similarita' coseno)
    - sentence-transformers (embedding locali per Answer Relevance)

Input atteso: lista JSON con campi:
    example_id, dataset, model_id, question, generated, context
Campi opzionali copiati se presenti:
    answers, is_correct_string

Output:
    <output-dir>/rag_metrics_details.json
    <output-dir>/rag_metrics_summary_by_model_dataset.json

Variabili d'ambiente richieste (da .env):
    LLM_API_URL     URL dell'endpoint OpenAI-compatible
    LLM_API_TOKEN   Token di autenticazione per l'API

Esempio di utilizzo:
    python llm_as_a_judge_gpt4o_rag_metrics_uniqa.py \\
      --input-file data/llm_as_a_judge/uniqa/ragass_samples_uniqa_it.json \\
      --output-dir data/llm_as_a_judge/uniqa_rag_metrics \\
      --judge-model gpt-4o-mini \\
      --concurrency 16 \\
      --datasets uniqa_it
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

# Importazione opzionale di sentence-transformers per gli embedding locali
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Caricamento delle variabili d'ambiente dal file .env
load_dotenv(dotenv_path=".env", override=True)

# Lettura delle credenziali API dall'ambiente
API_URL = os.getenv("LLM_API_URL", "").strip()
API_TOKEN = os.getenv("LLM_API_TOKEN", "").strip()

# Configurazione di default del giudice LLM e degli embedding
JUDGE_MODEL = "gpt-4o-mini"
CONCURRENCY = 4
N_QUESTIONS = 5
CTX_MAX_CHARS = 25000
EMBEDDING_MODEL = "BAAI/bge-m3"

# Limiti di token per le diverse metriche
MAX_TOKENS_FAITH = 300
MAX_TOKENS_AR_QGEN = 80
MAX_TOKENS_CR = 400

# Percorsi di default per input e output
DEFAULT_INPUT = Path("data/llm_as_a_judge/uniqa/ragass_samples_uniqa_it.json")
DEFAULT_OUTPUT_DIR = Path("data/llm_as_a_judge/uniqa_rag_metrics")
# Insieme dei dataset supportati da questo script
DATASETS = {"uniqa_it"}

# Variabili globali per concorrenza, modello di embedding e cache
sem = None
embedder = None
embed_lock = None
question_embedding_cache = {}


def _normalize_openai_base_url(url: str) -> str:
    """Normalizza l'URL base per il client AsyncOpenAI.

    Rimuove eventuali suffissi di endpoint (es. /chat/completions) e
    garantisce che l'URL termini con '/v1' come richiesto dal client OpenAI.

    Args:
        url: URL grezzo letto dalla variabile d'ambiente LLM_API_URL.

    Returns:
        URL normalizzato con suffisso '/v1'.

    Raises:
        RuntimeError: Se l'URL e' vuoto o non impostato.
    """
    if not url:
        raise RuntimeError("LLM_API_URL non impostata nel file .env o nell'ambiente.")
    url = url.rstrip("/")
    # Rimozione di eventuali suffissi di endpoint gia' presenti
    for suffix in ["/chat/completions", "/v1/chat/completions"]:
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    # Aggiunta del suffisso /v1 se non presente
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def _clean_api_key(token: str) -> str:
    """Pulisce il token API rimuovendo spazi e il prefisso 'Bearer' se presente.

    Args:
        token: Token grezzo letto dalla variabile d'ambiente LLM_API_TOKEN.

    Returns:
        Token pulito, pronto per l'autenticazione.

    Raises:
        RuntimeError: Se il token e' vuoto o non impostato.
    """
    if not token:
        raise RuntimeError("LLM_API_TOKEN non impostata nel file .env o nell'ambiente.")
    token = token.strip()
    # Rimozione del prefisso 'Bearer' se l'utente lo ha incluso nel token
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()
    return token


def make_client() -> AsyncOpenAI:
    """Crea e restituisce un client AsyncOpenAI configurato con le credenziali d'ambiente.

    Returns:
        Istanza di AsyncOpenAI pronta per effettuare chiamate API.

    Raises:
        RuntimeError: Se le credenziali API non sono configurate correttamente.
    """
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
    """Effettua una singola chiamata asincrona al modello LLM giudice.

    Invia un messaggio di sistema e un messaggio utente al modello,
    e restituisce il contenuto testuale della risposta. Rimuove eventuali
    tag </think> prodotti da modelli con ragionamento esplicito.

    Args:
        client: Client AsyncOpenAI configurato.
        system: Prompt di sistema che definisce il ruolo del giudice.
        user: Prompt utente con la domanda/valutazione da effettuare.
        max_tokens: Numero massimo di token nella risposta.
        temperature: Temperatura di campionamento (default 0.0 per determinismo).

    Returns:
        Contenuto testuale della risposta del modello, ripulito.
    """
    # Chiamata asincrona all'API di completamento chat
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
    # Rimozione di eventuali tag di ragionamento esplicito (</think>)
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


def count_sentences(text: str) -> int:
    """Conta il numero di frasi in un testo usando la punteggiatura come separatore.

    Divide il testo sui caratteri di fine frase (.!?) e conta i segmenti
    non vuoti. Restituisce almeno 1 anche per testi senza punteggiatura.

    Args:
        text: Testo di cui contare le frasi.

    Returns:
        Numero di frasi trovate (minimo 1).
    """
    sentences = re.split(r"[.!?]+", text or "")
    return max(1, len([s for s in sentences if s.strip()]))


# ─────────────────────────────────────────────────────────────────────────────
# Faithfulness (fedelta' al contesto)
# ─────────────────────────────────────────────────────────────────────────────

# Prompt di sistema in italiano per l'estrazione delle affermazioni fattuali
_SYS_FAITH_EXTRACT = (
    "Estrai affermazioni fattuali atomiche da risposte in italiano. "
    "Elenca ogni affermazione su una nuova riga iniziando con 'statement: '."
)

# Prompt di sistema in italiano per la verifica delle affermazioni contro il contesto
_SYS_FAITH_VERIFY = (
    "Verifica se le affermazioni sono supportate dal contesto. "
    "Per ogni affermazione dai una breve spiegazione, poi un verdetto. "
    "Termina con verdetti in questo formato esatto, uno per riga:\n"
    "verdict: Yes\nverdict: No"
)


def parse_statements(text: str) -> list[str]:
    """Analizza il testo prodotto dall'LLM ed estrae la lista di affermazioni fattuali.

    Cerca prima righe che iniziano con 'statement:', poi come fallback
    tenta il parsing di elenchi numerati o puntati.

    Args:
        text: Testo grezzo prodotto dall'LLM nella fase di estrazione.

    Returns:
        Lista di stringhe, ciascuna contenente un'affermazione fattuale.
    """
    statements = []
    # Prima strategia: ricerca righe con prefisso 'statement:'
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.lower().startswith("statement:"):
            s = line[len("statement:"):].strip()
            if s:
                statements.append(s)

    # Fallback: parsing di elenchi numerati (1. / 1)) o puntati (- )
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
    """Analizza il testo prodotto dall'LLM ed estrae i verdetti di supporto.

    Cerca prima righe con prefisso 'verdict:', poi come fallback conta
    le occorrenze di 'yes' e 'no' nel testo.

    Args:
        text: Testo grezzo prodotto dall'LLM nella fase di verifica.
        n: Numero atteso di verdetti (uno per affermazione).

    Returns:
        Lista di booleani di lunghezza n; True = supportata, False = non supportata.
    """
    verdicts = []
    # Ricerca righe con prefisso 'verdict:'
    for line in text.strip().split("\n"):
        line = line.strip().lower()
        if line.startswith("verdict:"):
            val = line[len("verdict:"):].strip()
            verdicts.append(val.startswith("yes"))

    # Fallback se il numero di verdetti trovati non corrisponde
    if len(verdicts) != n:
        yeses = len(re.findall(r"\byes\b", text.lower()))
        nos = len(re.findall(r"\bno\b", text.lower()))
        if yeses + nos:
            verdicts = [True] * min(yeses, n) + [False] * max(0, n - yeses)
        else:
            verdicts = [False] * n

    # Troncamento o padding a lunghezza n
    return (verdicts + [False] * n)[:n]


async def compute_faithfulness(client: AsyncOpenAI, question: str, generated: str, context: str) -> dict:
    """Calcola la metrica Faithfulness per un singolo record UniQA.

    La fedelta' misura la percentuale di affermazioni fattuali nella risposta
    generata che sono supportate dal contesto fornito. Segue un processo
    in due fasi con prompt in italiano: (1) estrazione delle affermazioni,
    (2) verifica contro il contesto.

    Args:
        client: Client AsyncOpenAI configurato.
        question: Domanda originale del dataset in italiano.
        generated: Risposta generata dal modello RAG.
        context: Contesto fornito al modello RAG.

    Returns:
        Dizionario con chiavi:
            - faithfulness (float): Rapporto affermazioni supportate / totale.
            - n_statements (int): Numero di affermazioni estratte.
            - n_verified (int): Numero di affermazioni supportate dal contesto.
            - _faith_statements_raw (str): Output grezzo della fase di estrazione.
            - _faith_verdicts_raw (str): Output grezzo della fase di verifica.
    """
    generated = str(generated or "")
    context = str(context or "")

    # Se la risposta generata e' vuota, la fedelta' e' per definizione 1.0
    if not generated.strip():
        return {
            "faithfulness": 1.0,
            "n_statements": 0,
            "n_verified": 0,
            "_faith_statements_raw": "",
            "_faith_verdicts_raw": "",
        }

    # Fase 1: Estrazione delle affermazioni fattuali dalla risposta (prompt in italiano)
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

    # Se non sono state estratte affermazioni, la fedelta' e' 1.0 per definizione
    if not statements:
        return {
            "faithfulness": 1.0,
            "n_statements": 0,
            "n_verified": 0,
            "_faith_statements_raw": raw_stmts,
            "_faith_verdicts_raw": "",
        }

    # Fase 2: Verifica di ogni affermazione contro il contesto (prompt in italiano)
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

    # Calcolo del punteggio di fedelta' come rapporto verificate/totale
    return {
        "faithfulness": round(n_verified / len(statements), 6),
        "n_statements": len(statements),
        "n_verified": n_verified,
        "_faith_statements_raw": raw_stmts,
        "_faith_verdicts_raw": raw_verds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Context Relevance (rilevanza del contesto)
# ─────────────────────────────────────────────────────────────────────────────

# Prompt di sistema in italiano per l'estrazione delle frasi rilevanti dal contesto
_SYS_CTXREL = (
    "Estrai dal contesto le frasi rilevanti per rispondere alla domanda. "
    "Output SOLO con frasi copiate testualmente, oppure 'Insufficient Information' se nessuna frase è rilevante."
)


async def compute_context_relevance(client: AsyncOpenAI, question: str, context: str) -> dict:
    """Calcola la metrica Context Relevance per un singolo record UniQA.

    La rilevanza del contesto misura la proporzione di frasi nel contesto
    che sono utili per rispondere alla domanda. Chiede all'LLM (con prompt
    in italiano) di copiare solo le frasi rilevanti e ne conta il rapporto
    rispetto al totale.

    Args:
        client: Client AsyncOpenAI configurato.
        question: Domanda originale del dataset in italiano.
        context: Contesto fornito al modello RAG.

    Returns:
        Dizionario con chiavi:
            - context_relevance (float): Rapporto frasi rilevanti / frasi totali.
            - n_extracted (int): Numero di frasi estratte come rilevanti.
            - n_total_sentences (int): Numero totale di frasi nel contesto.
            - _cr_raw (str): Output grezzo dell'LLM.
    """
    context = str(context or "")
    # Se il contesto e' vuoto, la rilevanza e' 0.0
    if not context.strip():
        return {"context_relevance": 0.0, "n_extracted": 0, "n_total_sentences": 0, "_cr_raw": ""}

    # Troncamento del contesto al limite massimo di caratteri
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
    # Conteggio delle frasi estratte (0 se la risposta indica informazione insufficiente)
    n_extracted = 0 if "insufficient information" in raw.lower() else count_sentences(raw)
    n_total = count_sentences(context_trunc)
    # Calcolo del punteggio limitato a 1.0
    score = min(1.0, n_extracted / n_total) if n_total > 0 else 0.0

    return {
        "context_relevance": round(score, 6),
        "n_extracted": n_extracted,
        "n_total_sentences": n_total,
        "_cr_raw": raw,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Answer Relevance: n domande sintetiche + similarita' coseno con embedding locali
# ─────────────────────────────────────────────────────────────────────────────

# Prompt di sistema per la generazione di domande sintetiche in italiano
_SYS_QGEN = (
    "Genera una domanda in italiano a cui la risposta data potrebbe rispondere. "
    "Output SOLO con la domanda, nient'altro."
)


async def generate_one_question(client: AsyncOpenAI, generated: str) -> str:
    """Genera una singola domanda sintetica in italiano a partire dalla risposta generata.

    Usata nella pipeline di Answer Relevance per creare domande che
    la risposta potrebbe plausibilmente soddisfare.

    Args:
        client: Client AsyncOpenAI configurato.
        generated: Risposta generata dal modello RAG.

    Returns:
        Testo della domanda sintetica in italiano generata dall'LLM.
    """
    gen_trunc = str(generated or "")[:800]
    user = (
        "Genera una domanda plausibile in italiano la cui risposta sarebbe il testo seguente. "
        "Mantieni la domanda specifica. Output solo con la domanda.\n\n"
        f"Risposta:\n{gen_trunc}"
    )
    return await llm_call(client, _SYS_QGEN, user, MAX_TOKENS_AR_QGEN, temperature=0.7)


def _clean_synthetic_question(q: str) -> str:
    """Pulisce una domanda sintetica rimuovendo prefissi numerati e virgolette.

    Args:
        q: Domanda sintetica grezza prodotta dall'LLM.

    Returns:
        Domanda pulita senza prefissi numerici o virgolette.
    """
    q = (q or "").strip()
    # Rimozione di prefissi numerati (es. "1. ", "2) ")
    q = re.sub(r"^[\d]+[.)]\s*", "", q).strip()
    # Rimozione di virgolette esterne
    q = q.strip('"').strip("'").strip()
    return q


async def embed_texts(texts: list[str]) -> np.ndarray:
    """Calcola gli embedding di una lista di testi usando il modello locale.

    L'operazione viene eseguita in un thread separato per non bloccare
    il loop di eventi, protetta da un lock per thread-safety.

    Args:
        texts: Lista di testi da codificare in vettori embedding.

    Returns:
        Array numpy con gli embedding normalizzati, uno per riga.
    """
    global embedder, embed_lock
    # Esecuzione dell'encoding in un thread separato con lock
    async with embed_lock:
        return await asyncio.to_thread(
            embedder.encode,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


async def get_question_embedding(question: str) -> np.ndarray:
    """Restituisce l'embedding della domanda originale, con cache.

    Se l'embedding e' gia' stato calcolato per la stessa domanda,
    viene restituito dalla cache senza ricalcolo.

    Args:
        question: Testo della domanda originale in italiano.

    Returns:
        Vettore numpy normalizzato dell'embedding della domanda.
    """
    key = question.strip()
    # Verifica se l'embedding e' gia' in cache
    cached = question_embedding_cache.get(key)
    if cached is not None:
        return cached
    # Calcolo dell'embedding e inserimento in cache
    emb = (await embed_texts([key]))[0]
    question_embedding_cache[key] = emb
    return emb


async def compute_answer_relevance(client: AsyncOpenAI, question: str, generated: str) -> dict:
    """Calcola la metrica Answer Relevance seguendo la pipeline RAGAS.

    Genera n domande sintetiche in italiano dalla risposta, calcola gli
    embedding della domanda originale e delle domande sintetiche, e
    restituisce la media della similarita' coseno come punteggio di rilevanza.

    Args:
        client: Client AsyncOpenAI configurato.
        question: Domanda originale del dataset in italiano.
        generated: Risposta generata dal modello RAG.

    Returns:
        Dizionario con chiavi:
            - answer_relevance (float): Media della similarita' coseno.
            - n_synthetic_questions (int): Numero di domande sintetiche valide.
            - synthetic_questions (list): Lista delle domande sintetiche pulite.
            - synthetic_question_scores (list): Punteggi individuali di similarita'.
            - _ar_qgen_raw (list): Risposte grezze dell'LLM per debug.
    """
    generated = str(generated or "")
    # Se la risposta generata e' vuota, la rilevanza e' 0.0
    if not generated.strip():
        return {
            "answer_relevance": 0.0,
            "n_synthetic_questions": 0,
            "synthetic_questions": [],
            "synthetic_question_scores": [],
            "_ar_qgen_raw": [],
        }

    # Generazione concorrente di n domande sintetiche in italiano
    q_tasks = [generate_one_question(client, generated) for _ in range(N_QUESTIONS)]
    raw_questions = await asyncio.gather(*q_tasks)

    # Pulizia e filtraggio delle domande sintetiche (minimo 5 caratteri)
    synthetic_questions = []
    for q in raw_questions:
        q = _clean_synthetic_question(q)
        if q and len(q) >= 5:
            synthetic_questions.append(q)

    # Troncamento al numero massimo di domande
    synthetic_questions = synthetic_questions[:N_QUESTIONS]
    if not synthetic_questions:
        return {
            "answer_relevance": 0.0,
            "n_synthetic_questions": 0,
            "synthetic_questions": [],
            "synthetic_question_scores": [],
            "_ar_qgen_raw": raw_questions,
        }

    # Calcolo degli embedding per domanda originale e domande sintetiche
    original_emb = await get_question_embedding(str(question or ""))
    synth_embs = await embed_texts(synthetic_questions)
    # Calcolo della similarita' coseno (i vettori sono gia' normalizzati)
    scores = [float(np.dot(original_emb, emb)) for emb in synth_embs]

    return {
        "answer_relevance": round(float(np.mean(scores)), 6),
        "n_synthetic_questions": len(synthetic_questions),
        "synthetic_questions": synthetic_questions,
        "synthetic_question_scores": [round(s, 6) for s in scores],
        "_ar_qgen_raw": raw_questions,
    }


def record_key(rec: dict) -> tuple:
    """Genera una chiave univoca per identificare un record.

    La chiave e' composta dalla tripla (dataset, model_id, example_id)
    e viene usata per il resume e la deduplicazione dei risultati.

    Args:
        rec: Dizionario del record.

    Returns:
        Tupla (dataset, model_id, example_id) che identifica univocamente il record.
    """
    return (rec.get("dataset"), rec.get("model_id"), rec.get("example_id"))


def is_valid_result(rec: dict) -> bool:
    """Verifica se un risultato di valutazione e' valido e completo.

    Un risultato e' valido solo se non ha errori e tutte e tre le metriche
    RAG (faithfulness, context_relevance, answer_relevance) sono presenti.

    Args:
        rec: Dizionario del risultato di valutazione.

    Returns:
        True se il risultato e' valido, False altrimenti.
    """
    if rec.get("error"):
        return False
    # Verifica che tutte le tre metriche siano presenti e non None
    required = ["faithfulness", "context_relevance", "answer_relevance"]
    return all(rec.get(k) is not None for k in required)


async def evaluate_record(client: AsyncOpenAI, rec: dict) -> dict:
    """Valuta un singolo record calcolando tutte e tre le metriche RAG.

    Acquisisce il semaforo di concorrenza, calcola in sequenza Faithfulness,
    Context Relevance e Answer Relevance, e gestisce eventuali errori.

    Args:
        client: Client AsyncOpenAI configurato.
        rec: Dizionario del record contenente question, generated, context, ecc.

    Returns:
        Dizionario con i metadati del record e i risultati di tutte le metriche.
    """
    async with sem:
        try:
            question = rec["question"]
            generated = rec.get("generated", "")
            context = rec.get("context", "")

            # Calcolo sequenziale delle tre metriche RAG
            faith = await compute_faithfulness(client, question, generated, context)
            cr = await compute_context_relevance(client, question, context)
            ar = await compute_answer_relevance(client, question, generated)
            error = None
        except Exception as e:
            # Gestione errore: risultati vuoti con campo error compilato
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
    """Carica i risultati gia' calcolati da un file di dettaglio preesistente.

    Implementa il meccanismo di resume: confronta i record target con quelli
    gia' presenti nel file di output, separando i record gia' completati
    con successo da quelli ancora da valutare.

    Args:
        details_path: Percorso del file JSON di dettaglio preesistente.
        target_records: Lista dei record target da valutare.

    Returns:
        Tupla (existing, pending) dove:
            - existing: lista dei record gia' valutati con successo.
            - pending: lista dei record ancora da valutare.
    """
    # Se il file non esiste, tutti i record sono da valutare
    if not details_path.exists():
        return [], target_records

    # Costruzione dell'insieme delle chiavi target
    target_keys = {record_key(r) for r in target_records}

    try:
        previous = json.loads(details_path.read_text(encoding="utf-8"))
    except Exception:
        # Se il file e' corrotto, si riparte da zero
        return [], target_records

    # Filtraggio dei record precedenti validi e appartenenti al target
    done = {}
    for r in previous:
        key = record_key(r)
        if key not in target_keys:
            continue
        # Esclusione dei record non validi (con errore o metriche mancanti)
        if not is_valid_result(r):
            continue
        done[key] = r

    # Separazione tra record completati e record pendenti mantenendo l'ordine originale
    existing = [done[record_key(r)] for r in target_records if record_key(r) in done]
    pending = [r for r in target_records if record_key(r) not in done]

    print(f"Resume: {len(existing)} record già validi, {len(pending)} da valutare.")
    return existing, pending


def build_summary(details: list) -> list:
    """Costruisce il riepilogo aggregato delle metriche RAG per modello e dataset.

    Raggruppa i risultati dettagliati per coppia (model_id, dataset) e
    calcola le medie di faithfulness, context_relevance e answer_relevance.

    Args:
        details: Lista dei risultati dettagliati di valutazione.

    Returns:
        Lista di dizionari con metriche aggregate per ogni coppia modello/dataset.
    """
    # Inizializzazione degli accumulatori per ogni coppia (model_id, dataset)
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

        # I record con errore vengono conteggiati separatamente
        if not is_valid_result(r):
            g["n_errors"] += 1
            continue

        # Accumulazione delle metriche per il calcolo delle medie
        g["n"] += 1
        g["faithfulness_sum"] += float(r["faithfulness"])
        g["context_relevance_sum"] += float(r["context_relevance"])
        g["answer_relevance_sum"] += float(r["answer_relevance"])
        g["n_statements_sum"] += int(r.get("n_statements", 0))
        g["n_verified_sum"] += int(r.get("n_verified", 0))
        g["n_synthetic_questions_sum"] += int(r.get("n_synthetic_questions", 0))

    # Costruzione della lista di riepilogo ordinata per dataset e modello
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
    """Esegue la valutazione asincrona di tutti i record pendenti per le metriche RAG.

    Crea il client API, lancia i task di valutazione con concorrenza controllata
    dal semaforo, salva checkpoint periodici e restituisce la lista completa
    dei risultati (esistenti + nuovi).

    Args:
        records: Lista dei record da valutare.
        details_path: Percorso del file JSON per il salvataggio progressivo.
        existing_details: Lista dei risultati gia' completati dal resume.

    Returns:
        Lista completa di tutti i risultati dettagliati (precedenti + nuovi).
    """
    # Creazione del client API
    client = make_client()

    print(f"Judge:          {JUDGE_MODEL}")
    print(f"Embedding:      {EMBEDDING_MODEL}")
    print(f"Record:         {len(records)} nuovi")
    print(f"Concurrency:    {CONCURRENCY}")
    print(f"AR n:           {N_QUESTIONS}")
    print(f"LLM calls:      circa {len(records) * (2 + 1 + N_QUESTIONS)} nuovi")
    print(f"API base:       {_normalize_openai_base_url(API_URL)}")
    print(f"CTX chars:      {CTX_MAX_CHARS}")

    # Creazione dei task asincroni per ogni record
    tasks = [evaluate_record(client, rec) for rec in records]
    details = list(existing_details)
    start_done = len(details)
    total = start_done + len(records)

    t0 = time.time()

    # Ciclo asincrono con barra di progresso
    async for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks)):
        details.append(await coro)

        # Salvataggio checkpoint ogni 100 record completati
        if (len(details) - start_done) % 100 == 0:
            details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Checkpoint: {len(details)}/{total}")

    # Salvataggio finale di tutti i risultati
    details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"Completato in {elapsed/60:.1f} min.")
    return details


def main():
    """Punto di ingresso principale dello script.

    Analizza gli argomenti da riga di comando, carica il modello di embedding
    locale, carica i record dal file di input, filtra per dataset e modelli
    richiesti, avvia la valutazione asincrona e salva i risultati su disco.
    """
    global JUDGE_MODEL, CONCURRENCY, N_QUESTIONS, CTX_MAX_CHARS, EMBEDDING_MODEL
    global sem, embedder, embed_lock

    # Definizione e parsing degli argomenti da riga di comando
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

    # Aggiornamento delle variabili globali con i valori da riga di comando
    JUDGE_MODEL = args.judge_model
    CONCURRENCY = args.concurrency
    N_QUESTIONS = args.n_questions
    CTX_MAX_CHARS = args.ctx_max_chars
    EMBEDDING_MODEL = args.embedding_model
    sem = asyncio.Semaphore(CONCURRENCY)
    embed_lock = asyncio.Lock()

    # Verifica che sentence-transformers sia installato
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers non installato. Installa con: pip install sentence-transformers")

    # Caricamento del modello di embedding locale
    print(f"Caricamento embedding model locale: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Configurazione dei percorsi di input e output
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    details_path = output_dir / "rag_metrics_details.json"
    summary_path = output_dir / "rag_metrics_summary_by_model_dataset.json"

    # Caricamento dei record dal file JSON di input
    records = json.loads(input_path.read_text(encoding="utf-8"))

    # Filtraggio dei record per i dataset richiesti
    wanted_datasets = set(args.datasets) if args.datasets else set(r.get("dataset") for r in records)
    records = [r for r in records if r.get("dataset") in wanted_datasets]

    # Filtraggio opzionale per modelli specifici
    if args.models:
        records = [r for r in records if r.get("model_id") in set(args.models)]

    # Limitazione opzionale del numero di campioni
    if args.n_samples:
        records = records[:args.n_samples]

    print(f"Input:   {input_path}")
    print(f"Output:  {output_dir}")
    print(f"Dataset: {sorted(wanted_datasets)}")
    print(f"Totale record target: {len(records)}")

    # Resume: caricamento dei risultati preesistenti e separazione dei record pendenti
    existing, pending = load_existing_details(details_path, records)
    # Esecuzione della valutazione asincrona
    details = asyncio.run(run(pending, details_path, existing))

    # Costruzione e salvataggio del riepilogo aggregato
    summary = build_summary(details)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Salvato dettaglio: {details_path}")
    print(f"Salvato summary:   {summary_path}")


if __name__ == "__main__":
    main()
