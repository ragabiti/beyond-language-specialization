"""
llm_as_a_judge_gpt4o_eng.py

Valutazione LLM-as-a-Judge della metrica Answer Correctness sui dataset inglesi
(triviaqa, nq, bioasq).

Il modulo utilizza un LLM giudice (per default GPT-4o) tramite endpoint
OpenAI-compatible per stabilire se la risposta generata da un modello RAG
e' fattualmente corretta rispetto alle risposte gold.

Dataset supportati:
    - triviaqa
    - nq
    - bioasq

Dipendenze principali:
    - openai (AsyncOpenAI per chiamate asincrone)
    - python-dotenv (caricamento variabili da .env)
    - asyncio (orchestrazione concorrente)
    - tqdm (barra di avanzamento asincrona)

Input atteso:
    data/ragass/ragass_samples_eng_updated.json

Campi richiesti per ogni record:
    example_id, dataset, model_id, question, answers, generated

Output:
    <output-dir>/answer_correctness_details.json
    <output-dir>/answer_correctness_accuracy_by_model_dataset.json

Formato output summary:
[
  {
    "model_id": "...",
    "dataset": "bioasq|nq|triviaqa",
    "accuracy": 0.0,
    "n": 333,
    "n_correct": 0,
    "n_errors": 0
  }
]

Variabili d'ambiente richieste (da .env):
    LLM_API_URL     URL dell'endpoint OpenAI-compatible
    LLM_API_TOKEN   Token di autenticazione per l'API

Esempi di utilizzo:
    python llm_as_a_judge_gpt4o_eng.py \\
      --input-file data/ragass/ragass_samples_eng_updated.json \\
      --output-dir data/ragass/results_gpt4o_eng_ac

    python llm_as_a_judge_gpt4o_eng.py \\
      --input-file data/ragass/ragass_samples_eng_updated.json \\
      --output-dir data/ragass/results_gpt4o_eng_ac \\
      --n-samples 50

    python llm_as_a_judge_gpt4o_eng.py \\
      --judge-model gpt-4o-mini
"""

import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

# Caricamento delle variabili d'ambiente dal file .env
load_dotenv(dotenv_path=".env", override=True)

# Lettura delle credenziali API dall'ambiente
API_URL = os.getenv("LLM_API_URL", "").strip()
API_TOKEN = os.getenv("LLM_API_TOKEN", "").strip()

# Configurazione di default del giudice LLM
JUDGE_MODEL = "gpt-4o"
CONCURRENCY = 4
MAX_TOKENS_AC = 20

# Percorsi di default per input e output
DEFAULT_INPUT = Path("data/ragass/ragass_samples_eng_updated.json")
DEFAULT_OUTPUT_DIR = Path("data/ragass/results_gpt4o_eng_ac")
# Insieme dei dataset inglesi supportati
DATASETS = {"triviaqa", "nq", "bioasq"}


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

    # Rimozione dello slash finale
    url = url.rstrip("/")
    # Rimozione di eventuali suffissi di endpoint gia' presenti
    suffixes = ["/chat/completions", "/v1/chat/completions"]
    for suffix in suffixes:
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


# Prompt di sistema per la valutazione della correttezza della risposta
_SYS_CORRECTNESS = (
    "You are a strict factual answer evaluator. "
    "Output ONLY one word: correct or incorrect."
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


def parse_correctness(raw: str) -> float:
    """Analizza la risposta grezza del giudice ed estrae il verdetto di correttezza.

    Prende la prima parola della risposta, la normalizza rimuovendo caratteri
    non alfabetici, e verifica se corrisponde a 'correct'.

    Args:
        raw: Risposta grezza del modello giudice.

    Returns:
        1.0 se la prima parola e' 'correct', 0.0 altrimenti.
    """
    if not raw.strip():
        return 0.0
    # Estrazione e normalizzazione della prima parola
    first_word = raw.lower().strip().split()[0]
    first_word = re.sub(r"[^a-z]", "", first_word)
    return float(first_word == "correct")


async def compute_answer_correctness(
    client: AsyncOpenAI,
    question: str,
    answers: list,
    generated: str,
) -> dict:
    """Calcola la metrica Answer Correctness per un singolo record.

    Costruisce il prompt di valutazione confrontando la risposta generata
    con le risposte gold, invia la richiesta al giudice LLM, e interpreta
    il verdetto.

    Args:
        client: Client AsyncOpenAI configurato.
        question: Domanda originale del dataset.
        answers: Lista delle risposte gold accettabili.
        generated: Risposta generata dal modello RAG da valutare.

    Returns:
        Dizionario con chiavi:
            - answer_correctness (float): 1.0 se corretta, 0.0 altrimenti.
            - _ac_raw (str): Risposta grezza del giudice per debug.
    """
    # Unione delle risposte gold separate da pipe
    gold = " | ".join(str(a) for a in answers)

    # Troncamento per evitare di superare i limiti di contesto
    gold_trunc = gold[:800]
    gen_trunc = str(generated)[:800]

    # Costruzione del prompt utente con domanda, risposte gold e risposta generata
    user = (
        f"Question: {question}\n"
        f"Gold answers (any one is acceptable): {gold_trunc}\n"
        f"Model answer: {gen_trunc}\n\n"
        "Is the model answer correct? "
        "The answer is correct if it contains the same factual information as any gold answer, "
        "even if phrased differently. Ignore minor wording differences. "
        "If the model answer is empty, evasive, unrelated, or contradicts the gold answer, it is incorrect.\n"
        "Output exactly one word: correct or incorrect."
    )

    # Invocazione del giudice LLM e parsing del verdetto
    raw = await llm_call(client, _SYS_CORRECTNESS, user, MAX_TOKENS_AC)
    return {
        "answer_correctness": parse_correctness(raw),
        "_ac_raw": raw,
    }


# Semaforo globale per il controllo della concorrenza (inizializzato in main)
sem = None


async def evaluate_record(client: AsyncOpenAI, rec: dict) -> dict:
    """Valuta un singolo record del dataset tramite il giudice LLM.

    Acquisisce il semaforo di concorrenza, invoca la valutazione di
    Answer Correctness, e gestisce eventuali errori restituendo un
    risultato con campo 'error' compilato.

    Args:
        client: Client AsyncOpenAI configurato.
        rec: Dizionario del record contenente question, answers, generated, ecc.

    Returns:
        Dizionario con i metadati del record e il risultato della valutazione.
    """
    async with sem:
        try:
            # Calcolo della correttezza della risposta
            ac = await compute_answer_correctness(
                client=client,
                question=rec["question"],
                answers=rec["answers"],
                generated=rec.get("generated", ""),
            )
            error = None
        except Exception as e:
            # Gestione errore: registrazione del tipo e messaggio dell'eccezione
            ac = {"answer_correctness": None, "_ac_raw": f"error: {type(e).__name__}: {str(e)[:160]}"}
            error = f"{type(e).__name__}: {str(e)[:160]}"

        return {
            "example_id": rec.get("example_id"),
            "dataset": rec.get("dataset"),
            "model_id": rec.get("model_id"),
            "answer_correctness": ac["answer_correctness"],
            "_ac_raw": ac["_ac_raw"],
            "error": error,
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
        # Esclusione dei record con correttezza non calcolata
        if r.get("answer_correctness") is None:
            continue
        # Esclusione dei record terminati con errore
        if str(r.get("_ac_raw", "")).startswith("error:"):
            continue
        done[key] = r

    # Separazione tra record completati e record pendenti mantenendo l'ordine originale
    existing = [done[record_key(r)] for r in target_records if record_key(r) in done]
    pending = [r for r in target_records if record_key(r) not in done]

    print(f"Resume: {len(existing)} record già validi, {len(pending)} da valutare.")
    return existing, pending


def build_summary(details: list) -> list:
    """Costruisce il riepilogo aggregato delle metriche per modello e dataset.

    Raggruppa i risultati dettagliati per coppia (model_id, dataset) e
    calcola l'accuratezza come rapporto tra risposte corrette e totale.

    Args:
        details: Lista dei risultati dettagliati di valutazione.

    Returns:
        Lista di dizionari con accuracy aggregata per ogni coppia modello/dataset.
    """
    # Raggruppamento dei conteggi per coppia (model_id, dataset)
    grouped = defaultdict(lambda: {"n": 0, "n_correct": 0, "n_errors": 0})

    for r in details:
        dataset = r.get("dataset")
        model_id = r.get("model_id")
        key = (model_id, dataset)

        # I record con errore vengono conteggiati separatamente
        if r.get("answer_correctness") is None:
            grouped[key]["n_errors"] += 1
            continue

        grouped[key]["n"] += 1
        grouped[key]["n_correct"] += int(float(r["answer_correctness"]) == 1.0)

    # Costruzione della lista di riepilogo ordinata per dataset e modello
    summary = []
    for (model_id, dataset), stats in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        n = stats["n"]
        acc = round(stats["n_correct"] / n, 6) if n else None
        summary.append({
            "model_id": model_id,
            "dataset": dataset,
            "accuracy": acc,
            "n": n,
            "n_correct": stats["n_correct"],
            "n_errors": stats["n_errors"],
        })

    return summary


async def run(records: list, details_path: Path, existing_details: list) -> list:
    """Esegue la valutazione asincrona di tutti i record pendenti.

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

    print(f"Judge:       {JUDGE_MODEL}")
    print(f"Record:      {len(records)} nuovi")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"API base:    {_normalize_openai_base_url(API_URL)}")

    # Creazione dei task asincroni per ogni record
    tasks = [evaluate_record(client, rec) for rec in records]
    details = list(existing_details)
    start_done = len(details)
    total = start_done + len(records)

    t0 = time.time()

    # Ciclo asincrono con barra di progresso
    async for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks)):
        details.append(await coro)

        # Salvataggio checkpoint ogni 250 record completati
        if (len(details) - start_done) % 250 == 0:
            details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Checkpoint: {len(details)}/{total}")

    # Salvataggio finale di tutti i risultati
    details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"Completato in {elapsed/60:.1f} min.")
    return details


def main():
    """Punto di ingresso principale dello script.

    Analizza gli argomenti da riga di comando, carica i record dal file di input,
    filtra per dataset e modelli richiesti, avvia la valutazione asincrona e
    salva i risultati (dettaglio e riepilogo) su disco.
    """
    global JUDGE_MODEL, CONCURRENCY, sem

    # Definizione e parsing degli argomenti da riga di comando
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--datasets", type=str, nargs="+", default=["triviaqa", "nq", "bioasq"])
    args = parser.parse_args()

    # Aggiornamento delle variabili globali con i valori da riga di comando
    JUDGE_MODEL = args.judge_model
    CONCURRENCY = args.concurrency
    sem = asyncio.Semaphore(CONCURRENCY)

    # Configurazione dei percorsi di input e output
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    details_path = output_dir / "answer_correctness_details.json"
    summary_path = output_dir / "answer_correctness_accuracy_by_model_dataset.json"

    # Caricamento dei record dal file JSON di input
    records = json.loads(input_path.read_text(encoding="utf-8"))

    # Filtraggio dei record per i dataset richiesti
    wanted_datasets = set(args.datasets)
    records = [r for r in records if r.get("dataset") in wanted_datasets]

    # Avviso se sono stati richiesti dataset non standard
    unknown = wanted_datasets - DATASETS
    if unknown:
        print(f"[WARN] Dataset non standard richiesti: {sorted(unknown)}")

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
