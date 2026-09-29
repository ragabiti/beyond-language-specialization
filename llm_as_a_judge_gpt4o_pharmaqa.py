"""
llm_as_a_judge_gpt4o_pharmaqa.py

Valutazione LLM-as-a-Judge della metrica Answer Correctness sul dataset
italiano PharmaQA (pharmaqa_it), specializzato in dominio farmaceutico/sanitario.

A differenza delle altre varianti, questo modulo utilizza ``aiohttp`` per
effettuare chiamate HTTP asincrone direttamente, senza passare dal client
AsyncOpenAI. I prompt di valutazione sono in italiano e specifici per il
dominio farmaceutico.

Dataset supportati:
    - pharmaqa_it

Dipendenze principali:
    - aiohttp (chiamate HTTP asincrone al backend LLM)
    - python-dotenv (caricamento variabili da .env)
    - asyncio (orchestrazione concorrente)
    - tqdm (barra di avanzamento asincrona)

Input atteso:
    data/llm_as_a_judge/pharmaqa/ragass_samples_pharmaqa.json

Campi richiesti per ogni record:
    example_id, dataset, model_id, question, answers, generated

Output:
    <output-dir>/answer_correctness_details.json
    <output-dir>/answer_correctness_accuracy_by_model_dataset.json

Variabili d'ambiente richieste (da .env):
    LLM_API_URL     URL dell'endpoint OpenAI-compatible
    LLM_API_TOKEN   Token di autenticazione per l'API

Esempio test:
    python llm_as_a_judge_gpt4o_pharmaqa.py \\
      --input-file data/llm_as_a_judge/pharmaqa/ragass_samples_pharmaqa.json \\
      --output-dir data/llm_as_a_judge/pharmaqa_test \\
      --judge-model gpt-4o-mini \\
      --concurrency 32 \\
      --datasets pharmaqa_it \\
      --n-samples 100

Esempio full:
    python llm_as_a_judge_gpt4o_pharmaqa.py \\
      --input-file data/llm_as_a_judge/pharmaqa/ragass_samples_pharmaqa.json \\
      --output-dir data/llm_as_a_judge/pharmaqa \\
      --judge-model gpt-4o-mini \\
      --concurrency 32 \\
      --datasets pharmaqa_it
"""

import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as atqdm

# Caricamento delle variabili d'ambiente dal file .env
load_dotenv(dotenv_path=".env", override=True)

# Lettura delle credenziali API dall'ambiente
API_URL = os.getenv("LLM_API_URL", "").strip()
API_TOKEN = os.getenv("LLM_API_TOKEN", "").strip()

# Configurazione di default del giudice LLM
JUDGE_MODEL = "gpt-4o-mini"
CONCURRENCY = 4
MAX_TOKENS_AC = 20
TIMEOUT_SEC = 120

# Percorsi di default per input e output
DEFAULT_INPUT = Path("data/llm_as_a_judge/pharmaqa/ragass_samples_pharmaqa.json")
DEFAULT_OUTPUT_DIR = Path("data/llm_as_a_judge/pharmaqa")
# Dataset di default da valutare
DEFAULT_DATASETS = ["pharmaqa_it"]


def _clean_auth_header(token: str) -> str:
    """Prepara il valore dell'header Authorization con prefisso Bearer.

    Se il token contiene gia' il prefisso 'Bearer', lo restituisce invariato.

    Args:
        token: Token grezzo letto dalla variabile d'ambiente LLM_API_TOKEN.

    Returns:
        Stringa nel formato 'Bearer <token>' pronta per l'header HTTP.

    Raises:
        RuntimeError: Se il token e' vuoto o non impostato.
    """
    if not token:
        raise RuntimeError("LLM_API_TOKEN non impostata nel file .env o nell'ambiente.")
    token = token.strip()
    # Se il prefisso 'Bearer' e' gia' presente, lo restituisce cosi' com'e'
    if token.lower().startswith("bearer "):
        return token
    return f"Bearer {token}"


def _check_api_url(url: str) -> str:
    """Verifica e normalizza l'URL dell'API rimuovendo lo slash finale.

    Args:
        url: URL grezzo letto dalla variabile d'ambiente LLM_API_URL.

    Returns:
        URL normalizzato senza slash finale.

    Raises:
        RuntimeError: Se l'URL e' vuoto o non impostato.
    """
    if not url:
        raise RuntimeError("LLM_API_URL non impostata nel file .env o nell'ambiente.")
    return url.rstrip("/")


# Prompt di sistema in italiano specializzato per il dominio farmaceutico
_SYS_CORRECTNESS = (
    "Sei un valutatore rigoroso di correttezza fattuale per risposte di ambito farmaceutico/sanitario. "
    "Devi produrre SOLO una parola: correct oppure incorrect. "
    "Non aggiungere spiegazioni."
)


async def llm_call(
    session: aiohttp.ClientSession,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    """Effettua una singola chiamata asincrona al modello LLM giudice tramite aiohttp.

    Invia un messaggio di sistema e un messaggio utente al modello,
    e restituisce il contenuto testuale della risposta. Rimuove eventuali
    tag </think> prodotti da modelli con ragionamento esplicito.

    Args:
        session: Sessione aiohttp gia' configurata con header di autenticazione.
        system: Prompt di sistema che definisce il ruolo del giudice.
        user: Prompt utente con la domanda/valutazione da effettuare.
        max_tokens: Numero massimo di token nella risposta.
        temperature: Temperatura di campionamento (default 0.0 per determinismo).

    Returns:
        Contenuto testuale della risposta del modello, ripulito.

    Raises:
        RuntimeError: Se la risposta HTTP ha codice di stato >= 400.
    """
    # Costruzione del payload JSON per l'API
    payload = {
        "model": JUDGE_MODEL,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    # Invio della richiesta POST asincrona
    async with session.post(_check_api_url(API_URL), json=payload) as resp:
        text = await resp.text()
        # Controllo del codice di stato HTTP
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
        data = json.loads(text)

    # Estrazione e pulizia del contenuto dalla risposta JSON
    content = (data["choices"][0]["message"].get("content") or "").strip()
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
    session: aiohttp.ClientSession,
    question: str,
    answers: list,
    generated: str,
) -> dict:
    """Calcola la metrica Answer Correctness per un singolo record PharmaQA.

    Costruisce il prompt di valutazione in italiano specializzato per il
    dominio farmaceutico, confrontando la risposta generata con le risposte
    gold, e interpreta il verdetto del giudice.

    Args:
        session: Sessione aiohttp configurata con header di autenticazione.
        question: Domanda originale del dataset in italiano (ambito farmaceutico).
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
    gold_trunc = gold[:1200]
    gen_trunc = str(generated)[:1200]

    # Costruzione del prompt utente in italiano per dominio farmaceutico
    user = (
        f"Domanda: {question}\n"
        f"Risposte corrette di riferimento (una qualunque è accettabile): {gold_trunc}\n"
        f"Risposta del modello: {gen_trunc}\n\n"
        "Valuta se la risposta del modello è fattualmente corretta rispetto alla risposta di riferimento. "
        "Considera corretta una risposta che contiene la stessa informazione clinica/farmaceutica essenziale, "
        "anche se formulata diversamente o più sinteticamente. "
        "Ignora differenze minori di stile, articoli, punteggiatura o sinonimi non sostanziali. "
        "Considera incorrect una risposta vuota, evasiva, non pertinente, contraddittoria, troppo generica, "
        "o che omette un elemento essenziale della risposta corretta.\n"
        "Output esattamente una parola: correct oppure incorrect."
    )

    # Invocazione del giudice LLM e parsing del verdetto
    raw = await llm_call(session, _SYS_CORRECTNESS, user, MAX_TOKENS_AC)
    return {
        "answer_correctness": parse_correctness(raw),
        "_ac_raw": raw,
    }


# Semaforo globale per il controllo della concorrenza (inizializzato in main)
sem = None


async def evaluate_record(session: aiohttp.ClientSession, rec: dict) -> dict:
    """Valuta un singolo record del dataset PharmaQA tramite il giudice LLM.

    Acquisisce il semaforo di concorrenza, invoca la valutazione di
    Answer Correctness, e gestisce eventuali errori restituendo un
    risultato con campo 'error' compilato.

    Args:
        session: Sessione aiohttp configurata con header di autenticazione.
        rec: Dizionario del record contenente question, answers, generated, ecc.

    Returns:
        Dizionario con i metadati del record e il risultato della valutazione.
    """
    async with sem:
        try:
            # Calcolo della correttezza della risposta
            ac = await compute_answer_correctness(
                session=session,
                question=rec["question"],
                answers=rec["answers"],
                generated=rec.get("generated", ""),
            )
            error = None
        except Exception as e:
            # Gestione errore: registrazione del tipo e messaggio dell'eccezione
            ac = {"answer_correctness": None, "_ac_raw": f"error: {type(e).__name__}: {str(e)[:200]}"}
            error = f"{type(e).__name__}: {str(e)[:200]}"

        return {
            "example_id": rec.get("example_id"),
            "dataset": rec.get("dataset", "pharmaqa_it"),
            "model_id": rec.get("model_id"),
            "answer_correctness": ac["answer_correctness"],
            "_ac_raw": ac["_ac_raw"],
            "error": error,
        }


def record_key(rec: dict) -> tuple:
    """Genera una chiave univoca per identificare un record.

    La chiave e' composta dalla tripla (dataset, model_id, example_id)
    e viene usata per il resume e la deduplicazione dei risultati.
    L'example_id viene convertito a stringa per uniformita'.

    Args:
        rec: Dizionario del record.

    Returns:
        Tupla (dataset, model_id, example_id) che identifica univocamente il record.
    """
    return (rec.get("dataset"), rec.get("model_id"), str(rec.get("example_id")))


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

    Crea una sessione aiohttp con header di autenticazione e timeout,
    lancia i task di valutazione con concorrenza controllata dal semaforo,
    salva checkpoint periodici e restituisce la lista completa dei risultati.

    Args:
        records: Lista dei record da valutare.
        details_path: Percorso del file JSON per il salvataggio progressivo.
        existing_details: Lista dei risultati gia' completati dal resume.

    Returns:
        Lista completa di tutti i risultati dettagliati (precedenti + nuovi).
    """
    # Configurazione degli header e del timeout per la sessione HTTP
    headers = {
        "Authorization": _clean_auth_header(API_TOKEN),
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_SEC)

    print(f"Judge:       {JUDGE_MODEL}")
    print(f"Record:      {len(records)} nuovi")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"API URL:     {_check_api_url(API_URL)}")

    details = list(existing_details)
    start_done = len(details)
    total = start_done + len(records)
    t0 = time.time()

    # Apertura della sessione aiohttp con contesto asincrono
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        # Creazione dei task asincroni per ogni record
        tasks = [evaluate_record(session, rec) for rec in records]

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
    parser.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS)
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
    # Identificazione dei dataset presenti nel file per informare l'utente
    present_datasets = sorted({str(r.get("dataset")) for r in records})
    print(f"Dataset presenti nel file: {present_datasets}")

    # Filtraggio dei record per i dataset richiesti
    wanted_datasets = set(args.datasets)
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
