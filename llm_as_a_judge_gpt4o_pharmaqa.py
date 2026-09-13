"""
llm_as_a_judge_gpt4o_pharmaqa.py

Valutazione LLM-as-a-judge SOLO per Answer Correctness su PharmaQA italiano.

Input atteso:
    data/llm_as_a_judge/pharmaqa/ragass_samples_pharmaqa.json

Campi richiesti per ogni record:
    example_id, dataset, model_id, question, answers, generated

Dataset atteso:
    pharmaqa_it

Output:
    <output-dir>/answer_correctness_details.json
    <output-dir>/answer_correctness_accuracy_by_model_dataset.json

Usa endpoint OpenAI-compatible da .env:
    LLM_API_URL
    LLM_API_TOKEN

Esempio test:
    python llm_as_a_judge_gpt4o_pharmaqa.py \
      --input-file data/llm_as_a_judge/pharmaqa/ragass_samples_pharmaqa.json \
      --output-dir data/llm_as_a_judge/pharmaqa_test \
      --judge-model gpt-4o-mini \
      --concurrency 32 \
      --datasets pharmaqa_it \
      --n-samples 100

Esempio full:
    python llm_as_a_judge_gpt4o_pharmaqa.py \
      --input-file data/llm_as_a_judge/pharmaqa/ragass_samples_pharmaqa.json \
      --output-dir data/llm_as_a_judge/pharmaqa \
      --judge-model gpt-4o-mini \
      --concurrency 32 \
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

load_dotenv(dotenv_path=".env", override=True)

API_URL = os.getenv("LLM_API_URL", "").strip()
API_TOKEN = os.getenv("LLM_API_TOKEN", "").strip()

JUDGE_MODEL = "gpt-4o-mini"
CONCURRENCY = 4
MAX_TOKENS_AC = 20
TIMEOUT_SEC = 120

DEFAULT_INPUT = Path("data/llm_as_a_judge/pharmaqa/ragass_samples_pharmaqa.json")
DEFAULT_OUTPUT_DIR = Path("data/llm_as_a_judge/pharmaqa")
DEFAULT_DATASETS = ["pharmaqa_it"]


def _clean_auth_header(token: str) -> str:
    if not token:
        raise RuntimeError("LLM_API_TOKEN non impostata nel file .env o nell'ambiente.")
    token = token.strip()
    if token.lower().startswith("bearer "):
        return token
    return f"Bearer {token}"


def _check_api_url(url: str) -> str:
    if not url:
        raise RuntimeError("LLM_API_URL non impostata nel file .env o nell'ambiente.")
    return url.rstrip("/")


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

    async with session.post(_check_api_url(API_URL), json=payload) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
        data = json.loads(text)

    content = (data["choices"][0]["message"].get("content") or "").strip()
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


def parse_correctness(raw: str) -> float:
    if not raw.strip():
        return 0.0
    first_word = raw.lower().strip().split()[0]
    first_word = re.sub(r"[^a-z]", "", first_word)
    return float(first_word == "correct")


async def compute_answer_correctness(
    session: aiohttp.ClientSession,
    question: str,
    answers: list,
    generated: str,
) -> dict:
    gold = " | ".join(str(a) for a in answers)
    gold_trunc = gold[:1200]
    gen_trunc = str(generated)[:1200]

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

    raw = await llm_call(session, _SYS_CORRECTNESS, user, MAX_TOKENS_AC)
    return {
        "answer_correctness": parse_correctness(raw),
        "_ac_raw": raw,
    }


sem = None


async def evaluate_record(session: aiohttp.ClientSession, rec: dict) -> dict:
    async with sem:
        try:
            ac = await compute_answer_correctness(
                session=session,
                question=rec["question"],
                answers=rec["answers"],
                generated=rec.get("generated", ""),
            )
            error = None
        except Exception as e:
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
    return (rec.get("dataset"), rec.get("model_id"), str(rec.get("example_id")))


def load_existing_details(details_path: Path, target_records: list) -> tuple[list, list]:
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
        if r.get("answer_correctness") is None:
            continue
        if str(r.get("_ac_raw", "")).startswith("error:"):
            continue
        done[key] = r

    existing = [done[record_key(r)] for r in target_records if record_key(r) in done]
    pending = [r for r in target_records if record_key(r) not in done]

    print(f"Resume: {len(existing)} record già validi, {len(pending)} da valutare.")
    return existing, pending


def build_summary(details: list) -> list:
    grouped = defaultdict(lambda: {"n": 0, "n_correct": 0, "n_errors": 0})

    for r in details:
        dataset = r.get("dataset")
        model_id = r.get("model_id")
        key = (model_id, dataset)

        if r.get("answer_correctness") is None:
            grouped[key]["n_errors"] += 1
            continue

        grouped[key]["n"] += 1
        grouped[key]["n_correct"] += int(float(r["answer_correctness"]) == 1.0)

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

    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        tasks = [evaluate_record(session, rec) for rec in records]

        async for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks)):
            details.append(await coro)

            if (len(details) - start_done) % 250 == 0:
                details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"Checkpoint: {len(details)}/{total}")

    details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"Completato in {elapsed/60:.1f} min.")
    return details


def main():
    global JUDGE_MODEL, CONCURRENCY, sem

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS)
    args = parser.parse_args()

    JUDGE_MODEL = args.judge_model
    CONCURRENCY = args.concurrency
    sem = asyncio.Semaphore(CONCURRENCY)

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    details_path = output_dir / "answer_correctness_details.json"
    summary_path = output_dir / "answer_correctness_accuracy_by_model_dataset.json"

    records = json.loads(input_path.read_text(encoding="utf-8"))
    present_datasets = sorted({str(r.get("dataset")) for r in records})
    print(f"Dataset presenti nel file: {present_datasets}")

    wanted_datasets = set(args.datasets)
    records = [r for r in records if r.get("dataset") in wanted_datasets]

    if args.models:
        records = [r for r in records if r.get("model_id") in set(args.models)]

    if args.n_samples:
        records = records[:args.n_samples]

    print(f"Input:   {input_path}")
    print(f"Output:  {output_dir}")
    print(f"Dataset: {sorted(wanted_datasets)}")
    print(f"Totale record target: {len(records)}")

    existing, pending = load_existing_details(details_path, records)
    details = asyncio.run(run(pending, details_path, existing))

    summary = build_summary(details)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Salvato dettaglio: {details_path}")
    print(f"Salvato summary:   {summary_path}")


if __name__ == "__main__":
    main()
