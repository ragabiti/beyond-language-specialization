"""
run_ragass_phase1_uniqa.py  —  Fase 1: valutazione RAGAS su dataset UniQA IT

Esegue le chiamate LLM necessarie per calcolare quattro metriche RAGAS
sul dataset UniQA in lingua italiana. Versione adattata dalla pipeline
inglese con gestione robusta degli errori e salvataggio progressivo.

Dataset supportati:
    - UniQA IT (italiano, domande universitarie)

Modello judge:
    - Qwen/Qwen3-8B  (thinking mode disabilitato via extra_body)

Differenze rispetto alla versione inglese (run_ragass_phase1_eng.py):
    - Input:  data/ragass/ragass_samples_uniqa_it.json
    - Output: data/ragass/interim_llm_results_uniqa_it.json
    - Concorrenza ridotta a 16 (vs 64 EN) per stabilita'
    - Try/except per record: un errore non blocca il run completo
    - Salvataggio progressivo ogni 250 record (checkpoint)
    - Answer Relevance: generazione domande sintetiche in italiano

Dipendenze principali:
    - openai (AsyncOpenAI): client asincrono per comunicare con il server vLLM
    - asyncio: orchestrazione concorrente delle chiamate LLM
    - tqdm (tqdm.asyncio): barra di avanzamento asincrona

Prerequisito:
    vllm serve Qwen/Qwen3-8B --port 8000 --dtype bfloat16 \\
        --gpu-memory-utilization 0.90 --max-model-len 4096 \\
        --max-num-seqs 64

Uso:
    python run_ragass_phase1_uniqa.py                  # run completo
    python run_ragass_phase1_uniqa.py --n-samples 10   # test rapido
    python run_ragass_phase1_uniqa.py --n-samples 50   # benchmark
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

# ─── Configurazione globale ───────────────────────────────────────────────────
VLLM_URL    = "http://localhost:8000/v1"       # Endpoint del server vLLM locale
JUDGE_MODEL = "Qwen/Qwen3-8B"                 # Modello judge per la valutazione

# Limiti di token massimi per ciascuna metrica
MAX_TOKENS_AC    = 20    # Answer Correctness: risposta binaria breve
MAX_TOKENS_FAITH = 300   # Faithfulness: estrazione statement e verifica
MAX_TOKENS_AR    = 80    # Answer Relevance: generazione domanda sintetica
MAX_TOKENS_CR    = 400   # Context Relevance: estrazione frasi rilevanti

N_QUESTIONS  = 5         # Numero di domande sintetiche per Answer Relevance
CONCURRENCY  = 16        # Concorrenza ridotta rispetto alla versione EN per stabilita'

# Nessun troncamento reale — Qwen3-8B ha max_position_embeddings=40960.
# Con vLLM a --max-model-len 32768 copriamo tutto il contesto UniQA (max ~25000 token).
CTX_MAX_CHARS = 25000

IN_PATH  = Path("data/ragass/ragass_samples_uniqa_it.json")     # File di input predefinito
OUT_PATH = Path("data/ragass/interim_llm_results_uniqa_it.json") # File di output predefinito

# Disabilita il thinking mode per Qwen3 tramite parametro extra_body
QWEN3_EXTRA = {"enable_thinking": False}

DATASET_NAME = "uniqa_it"  # Nome del dataset per i log


# ─── Client OpenAI asincrono ──────────────────────────────────────────────────
def make_client() -> AsyncOpenAI:
    """Crea e restituisce un client OpenAI asincrono configurato per il server vLLM locale.

    La chiave API e' impostata a 'dummy' perche' vLLM non richiede autenticazione.

    Returns:
        AsyncOpenAI: istanza del client asincrono pronta per le chiamate.
    """
    return AsyncOpenAI(api_key="dummy", base_url=VLLM_URL)


async def llm_call(
    client: AsyncOpenAI,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    """Esegue una singola chiamata asincrona al modello judge tramite l'API chat completions.

    Invia un messaggio di sistema e uno utente al modello, e restituisce il testo
    generato. Gestisce automaticamente il fallback nel caso in cui il thinking mode
    di Qwen3 sia ancora attivo, rimuovendo il blocco <think>...</think>.

    Args:
        client: client AsyncOpenAI connesso al server vLLM.
        system: messaggio di sistema (prompt di contesto per il judge).
        user: messaggio utente (contiene la richiesta specifica).
        max_tokens: numero massimo di token nella risposta generata.
        temperature: temperatura di campionamento (0.0 = deterministico).

    Returns:
        Stringa con il testo generato dal modello, ripulito da spazi e
        da eventuali blocchi di thinking residui.
    """
    # Chiamata asincrona all'API chat completions del server vLLM
    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=QWEN3_EXTRA,
    )
    content = resp.choices[0].message.content.strip()
    # Fallback: se il thinking mode e' ancora attivo, estrai solo il testo dopo </think>
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


# ─────────────────────────────────────────────────────────────────────────────
# METRICA 1: Answer Correctness — valutazione binaria correttezza risposta
# ─────────────────────────────────────────────────────────────────────────────
_SYS_CORRECTNESS = (
    "/no_think\n"
    "You are a factual answer evaluator. "
    "Output ONLY one word: correct or incorrect."
)

async def compute_answer_correctness(
    client, question: str, answers: list, generated: str
) -> dict:
    """Calcola la metrica Answer Correctness confrontando la risposta generata con le gold answer.

    Il judge valuta se la risposta del modello contiene le stesse informazioni fattuali
    di almeno una delle risposte gold. Il prompt specifica che la risposta puo' essere
    in italiano e che va valutato il significato, non la lingua.

    Args:
        client: client AsyncOpenAI per la chiamata al judge.
        question: domanda originale del dataset.
        answers: lista di risposte gold accettabili.
        generated: risposta generata dal modello sotto valutazione.

    Returns:
        Dizionario con 'answer_correctness' (float: 0.0 o 1.0) e
        '_ac_raw' (stringa grezza della risposta del judge per debug).
    """
    # Concatena le risposte gold separate da pipe, convertendo in stringa per sicurezza
    gold = " | ".join(str(a) for a in answers)
    # Tronca gold e generated per evitare overflow del contesto
    gold_trunc = gold[:500] if len(gold) > 500 else gold
    gen_trunc  = generated[:500] if len(generated) > 500 else generated
    user = (
        f"Question: {question}\n"
        f"Gold answers (any one is acceptable): {gold_trunc}\n"
        f"Model answer: {gen_trunc}\n\n"
        "Is the model answer correct? An answer is correct if it contains "
        "the same factual information as any gold answer, even if phrased differently. "
        "The answer may be in Italian — evaluate the meaning, not the language.\n"
        "Output exactly one word: correct or incorrect."
    )
    raw        = await llm_call(client, _SYS_CORRECTNESS, user, MAX_TOKENS_AC)
    # Estrai la prima parola e normalizzala rimuovendo caratteri non alfabetici
    first_word = re.sub(r'[^a-z]', '', raw.lower().split()[0]) if raw.strip() else ""
    # Converti in valore binario: 1.0 se "correct", 0.0 altrimenti
    binary     = float(first_word == "correct")
    return {"answer_correctness": binary, "_ac_raw": raw}


# ─────────────────────────────────────────────────────────────────────────────
# METRICA 2: Faithfulness — verifica aderenza al contesto
# ─────────────────────────────────────────────────────────────────────────────
# Prompt di sistema per l'estrazione delle affermazioni fattuali
_SYS_FAITH_EXTRACT = (
    "/no_think\n"
    "You extract factual statements from answers. "
    "List each statement on a new line starting with 'statement: '."
)

# Prompt di sistema per la verifica delle affermazioni contro il contesto
_SYS_FAITH_VERIFY = (
    "/no_think\n"
    "You verify if statements are supported by a context. "
    "For each statement give a brief explanation, then a verdict. "
    "End with verdicts in this format — one per line:\n"
    "verdict: Yes\nverdict: No"
)

def parse_statements(text: str) -> list:
    """Estrae le affermazioni fattuali dal testo generato dal judge.

    Tenta prima il parsing nel formato atteso 'statement: ...'.
    Se non trova risultati, usa un fallback con formati alternativi
    (liste numerate o con trattino).

    Args:
        text: testo grezzo restituito dal judge contenente le affermazioni.

    Returns:
        Lista di stringhe, ciascuna rappresentante un'affermazione fattuale.
    """
    statements = []
    # Primo tentativo: formato standard 'statement: ...'
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.lower().startswith("statement:"):
            s = line[len("statement:"):].strip()
            if s:
                statements.append(s)
    # Fallback: se il judge non ha usato il formato standard, prova liste numerate o puntate
    if not statements:
        for line in text.strip().split("\n"):
            line = line.strip()
            m = re.match(r'^[\d]+[.)]\s+(.+)$', line)
            if m:
                statements.append(m.group(1).strip())
            elif line.startswith("- ") and len(line) > 2:
                statements.append(line[2:].strip())
    return statements

def parse_verdicts(text: str, n: int) -> list:
    """Estrae i verdetti (Yes/No) dal testo di verifica del judge.

    Cerca le righe nel formato 'verdict: Yes/No'. Se il numero di verdetti
    trovati non corrisponde al numero atteso di statement, applica un
    fallback contando le occorrenze della parola 'yes' nel testo.

    Args:
        text: testo grezzo del judge contenente i verdetti di verifica.
        n: numero atteso di verdetti (uno per ogni statement estratto).

    Returns:
        Lista di booleani di lunghezza n, dove True indica che lo
        statement e' supportato dal contesto.
    """
    verdicts = []
    # Parsing delle righe nel formato 'verdict: Yes/No'
    for line in text.strip().split("\n"):
        line = line.strip().lower()
        if line.startswith("verdict:"):
            val = line[len("verdict:"):].strip()
            verdicts.append(val.startswith("yes"))
    # Fallback: se il conteggio non corrisponde, conta le occorrenze di 'yes'
    if len(verdicts) != n:
        yeses = len(re.findall(r'\byes\b', text.lower()))
        verdicts = [True] * min(yeses, n) + [False] * max(0, n - yeses)
    # Tronca o riempi con False per garantire esattamente n verdetti
    return (verdicts + [False] * n)[:n]

async def compute_faithfulness(
    client, question: str, generated: str, context: str
) -> dict:
    """Calcola la metrica Faithfulness in due passaggi LLM.

    Passaggio 1: estrae le affermazioni fattuali dalla risposta generata.
    Passaggio 2: verifica ciascuna affermazione contro il contesto fornito.
    Il punteggio finale e' la frazione di affermazioni supportate dal contesto.
    Nota: la risposta generata viene troncata a 400 caratteri per l'estrazione.

    Args:
        client: client AsyncOpenAI per le chiamate al judge.
        question: domanda originale del dataset.
        generated: risposta generata dal modello sotto valutazione.
        context: contesto di riferimento recuperato dal sistema RAG.

    Returns:
        Dizionario con 'faithfulness' (float 0.0-1.0), 'n_statements'
        (numero di affermazioni estratte) e 'n_verified' (numero di
        affermazioni supportate dal contesto).
    """
    # Se la risposta generata e' vuota, restituisci faithfulness perfetto
    if not generated.strip():
        return {"faithfulness": 1.0, "n_statements": 0, "n_verified": 0}

    # Passaggio 1: estrazione delle affermazioni fattuali (risposta troncata a 400 chars)
    gen_trunc = generated[:400] if len(generated) > 400 else generated
    user_extract = (
        "Given a question and answer, create one or more statements "
        "from each sentence in the given answer.\n"
        f"question: {question}\n"
        f"answer: {gen_trunc}\n\n"
        "List each statement starting with 'statement: '"
    )
    raw_stmts  = await llm_call(client, _SYS_FAITH_EXTRACT, user_extract, MAX_TOKENS_FAITH)
    statements = parse_statements(raw_stmts)

    # Se nessuna affermazione estratta, restituisci faithfulness perfetto
    if not statements:
        return {"faithfulness": 1.0, "n_statements": 0, "n_verified": 0}

    # Passaggio 2: verifica di ogni affermazione contro il contesto
    stmts_str     = "\n".join(f"statement: {s}" for s in statements)
    # Tronca il contesto per rispettare il limite di token del modello
    context_trunc = context[:CTX_MAX_CHARS] if len(context) > CTX_MAX_CHARS else context
    user_verify = (
        "Consider the given context and following statements, then determine "
        "whether they are supported by the information present in the context. "
        "Provide a brief explanation for each statement before arriving at the "
        "verdict (Yes/No). Provide a final verdict for each statement in order "
        "at the end in the given format. Do not deviate from the specified format.\n\n"
        f"context: {context_trunc}\n\n"
        f"{stmts_str}\n\n"
        "End with verdicts:\nverdict: Yes\nverdict: No\n(one per statement)"
    )
    raw_verds = await llm_call(client, _SYS_FAITH_VERIFY, user_verify, MAX_TOKENS_FAITH)
    verdicts  = parse_verdicts(raw_verds, len(statements))

    # Calcolo dello score: rapporto tra affermazioni verificate e totali
    n_verified = sum(verdicts)
    return {
        "faithfulness": round(n_verified / len(statements), 4),
        "n_statements": len(statements),
        "n_verified":   n_verified,
    }


# ─────────────────────────────────────────────────────────────────────────────
# METRICA 3: Answer Relevance — pertinenza della risposta
# ─────────────────────────────────────────────────────────────────────────────
# Prompt di sistema per la generazione di domande sintetiche in italiano
_SYS_QGEN = "/no_think\nGeneri domande in italiano. Scrivi SOLO la domanda, nient'altro."

async def generate_one_question(client, generated: str) -> str:
    """Genera una singola domanda sintetica in italiano a partire dalla risposta del modello.

    Chiede al judge di inferire quale domanda in italiano avrebbe prodotto la
    risposta data. Usa temperatura 0.7 per ottenere diversita' tra le domande.
    La risposta viene troncata a 300 caratteri per rispettare il limite di token.

    Args:
        client: client AsyncOpenAI per la chiamata al judge.
        generated: risposta generata dal modello sotto valutazione.

    Returns:
        Stringa contenente la domanda sintetica generata in italiano.
    """
    # Tronca la risposta generata per evitare overflow del contesto
    gen_trunc = generated[:300] if len(generated) > 300 else generated
    user = f"Quale domanda ha come risposta: {gen_trunc}?"
    return await llm_call(client, _SYS_QGEN, user, MAX_TOKENS_AR, temperature=0.7)

async def compute_answer_relevance_llm(client, generated: str) -> dict:
    """Genera N_QUESTIONS domande sintetiche in italiano per il calcolo dell'Answer Relevance.

    Per ogni risposta generata, produce N_QUESTIONS domande sintetiche in parallelo.
    Le domande vengono poi pulite da formattazione indesiderata. Il calcolo finale
    della similarita' coseno avviene nella fase 2 (embedding).

    Args:
        client: client AsyncOpenAI per le chiamate al judge.
        generated: risposta generata dal modello sotto valutazione.

    Returns:
        Dizionario con 'synthetic_questions' (lista di stringhe con le domande
        sintetiche generate in italiano, massimo N_QUESTIONS).
    """
    # Se la risposta e' vuota, non generare domande sintetiche
    if not generated.strip():
        return {"synthetic_questions": []}
    # Lancia N_QUESTIONS chiamate LLM in parallelo con asyncio.gather
    tasks     = [generate_one_question(client, generated) for _ in range(N_QUESTIONS)]
    questions = await asyncio.gather(*tasks)
    # Pulizia delle domande generate: rimuovi formattazione indesiderata
    clean = []
    for q in questions:
        q = q.strip()
        # Scarta domande vuote, troppo corte o con formattazione markdown
        if not q or len(q) < 5 or q.startswith(("*", "-", "**")):
            continue
        # Rimuovi numerazione iniziale (es. "1. ", "2) ")
        q = re.sub(r'^[\d]+[.)]\s*', '', q).strip()
        clean.append(q)
    return {"synthetic_questions": clean[:N_QUESTIONS]}


# ─────────────────────────────────────────────────────────────────────────────
# METRICA 4: Context Relevance — rilevanza del contesto recuperato
# ─────────────────────────────────────────────────────────────────────────────
# Prompt di sistema per l'estrazione delle frasi rilevanti dal contesto
_SYS_CTXREL = (
    "/no_think\n"
    "You extract relevant sentences from a context. "
    "Output ONLY the extracted sentences verbatim, "
    "or 'Insufficient Information' if none are relevant."
)

def count_sentences(text: str) -> int:
    """Conta il numero di frasi in un testo, separandole per segni di punteggiatura.

    Usa come delimitatori punto, punto esclamativo e punto interrogativo.
    Restituisce almeno 1 per evitare divisioni per zero nel calcolo dello score.

    Args:
        text: testo di cui contare le frasi.

    Returns:
        Numero intero di frasi trovate (minimo 1).
    """
    sentences = re.split(r'[.!?]+', text)
    return max(1, len([s for s in sentences if s.strip()]))

async def compute_context_relevance(client, question: str, context: str) -> dict:
    """Calcola la metrica Context Relevance misurando la proporzione di frasi rilevanti.

    Chiede al judge di estrarre dal contesto solo le frasi utili a rispondere
    alla domanda. Lo score e' il rapporto tra frasi estratte e frasi totali
    nel contesto, limitato superiormente a 1.0.

    Args:
        client: client AsyncOpenAI per la chiamata al judge.
        question: domanda originale del dataset.
        context: contesto di riferimento recuperato dal sistema RAG.

    Returns:
        Dizionario con 'context_relevance' (float 0.0-1.0),
        'n_extracted' (frasi rilevanti estratte) e
        'n_total_sentences' (frasi totali nel contesto).
    """
    # Se il contesto e' vuoto, restituisci score zero
    if not context.strip():
        return {"context_relevance": 0.0, "n_extracted": 0, "n_total_sentences": 0}
    # Tronca il contesto per rispettare il limite di token
    context_trunc = context[:CTX_MAX_CHARS] if len(context) > CTX_MAX_CHARS else context
    user = (
        "From the context below, copy the sentences that help answer "
        "the question. If none, write 'Insufficient Information'. "
        "Do not modify any sentence.\n\n"
        f"Question: {question}\n"
        f"Context: {context_trunc}\n\n"
        "Output: relevant sentences only, or 'Insufficient Information'."
    )
    raw = await llm_call(client, _SYS_CTXREL, user, MAX_TOKENS_CR)
    # Conta le frasi estratte; se il judge dice "insufficient information", conta 0
    n_extracted = 0 if "insufficient information" in raw.lower() else count_sentences(raw)
    n_total     = count_sentences(context_trunc)
    # Calcola lo score come rapporto frasi estratte / frasi totali, max 1.0
    score       = min(1.0, n_extracted / n_total) if n_total > 0 else 0.0
    return {
        "context_relevance":  round(score, 4),
        "n_extracted":        n_extracted,
        "n_total_sentences":  n_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Controllo thinking mode — verifica che Qwen3 non generi reasoning
# ─────────────────────────────────────────────────────────────────────────────
def check_thinking_mode(results: list) -> bool:
    """Controlla se il thinking mode di Qwen3 e' ancora attivo nei risultati.

    Analizza i primi 5 record cercando parole chiave tipiche del reasoning
    interno del modello (es. 'thinking process', '<think>') nel campo '_ac_raw'
    della metrica Answer Correctness. Se trovate, significa che il flag
    enable_thinking=False non ha funzionato e i risultati potrebbero essere
    compromessi dal testo di reasoning.

    Args:
        results: lista dei risultati della valutazione.

    Returns:
        True se il thinking mode e' ancora attivo, False altrimenti.
    """
    thinking_keywords = ["thinking process", "<think>", "analyze the request"]
    # Controlla solo i primi 5 record come campione rappresentativo
    for rec in results[:5]:
        raw = rec.get("_ac_raw", "").lower()
        if any(kw in raw for kw in thinking_keywords):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione singolo record — con try/except per robustezza
# ─────────────────────────────────────────────────────────────────────────────
# Semaforo asincrono per limitare la concorrenza delle chiamate LLM
sem = asyncio.Semaphore(CONCURRENCY)

async def evaluate_record(client, rec: dict) -> dict:
    """Valuta un singolo record calcolando tutte e quattro le metriche RAGAS.

    Usa un semaforo asincrono per limitare il numero di valutazioni concorrenti.
    A differenza della versione inglese, include un try/except per gestire
    errori di singoli record senza interrompere l'intero run. In caso di errore,
    le metriche vengono impostate a None per essere facilmente identificabili.

    Args:
        client: client AsyncOpenAI per le chiamate al judge.
        rec: dizionario del record da valutare, con campi 'question',
             'answers', 'generated', 'context', ecc.

    Returns:
        Dizionario con i dati originali del record e i risultati delle
        quattro metriche RAGAS (o None in caso di errore).
    """
    # Il semaforo limita le valutazioni concorrenti per non sovraccaricare vLLM
    async with sem:
        try:
            # Metrica 1: Answer Correctness (1 chiamata LLM)
            ac     = await compute_answer_correctness(
                client, rec["question"], rec["answers"], rec["generated"]
            )
            # Metrica 2: Faithfulness (2 chiamate LLM)
            faith  = await compute_faithfulness(
                client, rec["question"], rec["generated"], rec["context"]
            )
            # Metrica 3: Answer Relevance — solo generazione domande (5 chiamate LLM)
            ar_llm = await compute_answer_relevance_llm(client, rec["generated"])
            # Metrica 4: Context Relevance (1 chiamata LLM)
            cr     = await compute_context_relevance(
                client, rec["question"], rec["context"]
            )
        except Exception as e:
            # Gestione errore: logga il warning e imposta le metriche a None
            print(f"\n[WARN] Record {rec.get('example_id','?')} ({rec.get('model_id','?').split('/')[-1]}) fallito: {type(e).__name__}: {str(e)[:80]}")
            ac     = {"answer_correctness": None, "_ac_raw": f"error: {type(e).__name__}"}
            faith  = {"faithfulness": None, "n_statements": 0, "n_verified": 0}
            ar_llm = {"synthetic_questions": []}
            cr     = {"context_relevance": None, "n_extracted": 0, "n_total_sentences": 0}

    # Assembla il risultato unendo dati originali e metriche calcolate
    return {
        "example_id":        rec["example_id"],
        "dataset":           rec["dataset"],
        "model_id":          rec["model_id"],
        "question":          rec["question"],
        "generated":         rec["generated"],
        "answers":           rec["answers"],
        "context":           rec["context"],
        "is_correct_string": rec["is_correct_string"],
        **ac, **faith, **ar_llm, **cr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
async def run(records: list, out_path: Path) -> list:
    """Esegue la valutazione RAGAS su tutti i record con salvataggio progressivo.

    Crea il client OpenAI, lancia le valutazioni concorrenti con barra di
    avanzamento e salva un checkpoint ogni 250 record. Al termine, verifica
    che il thinking mode sia disabilitato e conta i record con errori.

    Args:
        records: lista di dizionari con i record da valutare.
        out_path: percorso del file di output per il salvataggio progressivo.

    Returns:
        Lista di dizionari con i risultati della valutazione per ogni record.
    """
    client   = make_client()
    n_calls  = len(records) * 9
    n_models  = len(set(r["model_id"] for r in records))
    n_samples = len(set(r["example_id"] for r in records))

    # Riepilogo configurazione prima dell'esecuzione
    print(f"\n  Dataset:      {DATASET_NAME}")
    print(f"  Record:       {len(records)}  ({n_models} modelli × {n_samples} sample)")
    print(f"  LLM calls:    {n_calls}  (9/record: 1 AC + 2 Faith + {N_QUESTIONS} AR + 1 CR)")
    print(f"  Concurrency:  {CONCURRENCY}")
    print(f"  Judge:        {JUDGE_MODEL}")
    print(f"  Context max chars: {CTX_MAX_CHARS}")
    print(f"  Thinking:     disabled via extra_body\n")

    t0      = time.time()
    # Crea un task asincrono per ogni record
    tasks   = [evaluate_record(client, rec) for rec in records]
    results = []

    # Esegui tutti i task con barra di avanzamento asincrona
    async for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await coro)

        # Salvataggio progressivo (checkpoint) ogni 250 record completati
        if len(results) % 250 == 0:
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
            elapsed_so_far = time.time() - t0
            print(f"\n  Checkpoint: {len(results)}/{len(records)} record  "
                  f"({elapsed_so_far/60:.1f} min)")

    elapsed = time.time() - t0
    sec_per = elapsed / len(results) if results else 0

    # Verifica post-esecuzione: controlla che il thinking mode sia disabilitato
    if check_thinking_mode(results):
        print("\n⚠️  ATTENZIONE: thinking mode ancora attivo!")
    else:
        print(f"\n✓ Thinking mode: disabilitato correttamente")

    # Conta e segnala i record che hanno generato errori durante la valutazione
    errors = sum(1 for r in results if r.get("_ac_raw", "").startswith("error:"))
    if errors:
        print(f"⚠️  Record con errori: {errors}/{len(results)}")

    # Statistiche temporali
    print(f"✓ Completato in {elapsed/60:.1f} min  ({sec_per:.2f} sec/record)")
    return results


def main():
    """Punto di ingresso principale: parsing argomenti CLI, caricamento dati e avvio valutazione.

    Gestisce il flusso completo della Fase 1 per UniQA IT: legge il file JSON di input,
    filtra opzionalmente per modelli, esegue la valutazione RAGAS asincrona con
    salvataggio progressivo e salva i risultati intermedi.
    """
    # Definizione degli argomenti da riga di comando
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=None,
                        help="N record (es: 10 per test, 50 per benchmark)")
    parser.add_argument("--input-file", type=str, default=None,
                        help=f"File JSON di input (default: {IN_PATH})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Cartella di output (default: cartella del file input)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Limita la valutazione a questi model_id")
    args = parser.parse_args()

    # Risoluzione dei percorsi di input e output
    in_path  = Path(args.input_file) if args.input_file else IN_PATH
    out_dir  = Path(args.output_dir) if args.output_dir else in_path.parent
    out_path = out_dir / "interim_llm_results.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Caricamento del file JSON con tutti i record
    print(f"Caricamento: {in_path}")
    with open(in_path, encoding="utf-8") as f:
        records = json.load(f)

    # Filtro opzionale per modelli specifici
    if args.models:
        records = [r for r in records if r["model_id"] in args.models]
        print(f"Filtro modelli: {args.models}  ({len(records)} record)")

    # Sottocampionamento opzionale per test rapidi o benchmark
    if args.n_samples:
        records = records[:args.n_samples]
        print(f"Test/Benchmark: {len(records)} record")
    else:
        print(f"Record totali: {len(records)}")

    # Avvio della valutazione asincrona con salvataggio progressivo
    results = asyncio.run(run(records, out_path))

    # Salvataggio finale dei risultati intermedi su file JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Salvato: {out_path}  ({len(results)} record)")
    print(f"  Ora spegni vLLM e lancia: python run_ragass_phase2_uniqa.py --input-file {out_path} --output-dir {out_dir}")


if __name__ == "__main__":
    main()
