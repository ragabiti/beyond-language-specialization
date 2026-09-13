"""
llm_as_a_judge_gpt4o_uniqa.py

Valutazione LLM-as-a-judge SOLO per Answer Correctness su UniQA italiano.

Input atteso:
    data/llm_as_a_judge/uniqa/ragass_samples_uniqa_it.json

Campi richiesti per ogni record:
    example_id, dataset, model_id, question, answers, generated

Output:
    <output-dir>/answer_correctness_details.json
    <output-dir>/answer_correctness_accuracy_by_model_dataset.json

Usa endpoint OpenAI-compatible da .env:
    LLM_API_URL       es. https://portal.aiwave.ai/llm/api/v1/chat/completions
    LLM_API_TOKEN

Esempio test:
    python llm_as_a_judge_gpt4o_uniqa.py \
      --input-file data/llm_as_a_judge/uniqa/ragass_samples_uniqa_it.json \
      --output-dir data/llm_as_a_judge/uniqa_test \
      --judge-model gpt-4o-mini \
      --concurrency 8 \
      --n-samples 50

Esempio full:
    python llm_as_a_judge_gpt4o_uniqa.py \
      --input-file data/llm_as_a_judge/uniqa/ragass_samples_uniqa_it.json \
      --output-dir data/llm_as_a_judge/uniqa \
      --judge-model gpt-4o-mini \
      --concurrency 16 \
      --datasets uniqa
"""

import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as atqdm

load_dotenv(dotenv_path=".env", override=True)

API_URL = os.getenv("LLM_API_URL", "").strip()
API_TOKEN = os.getenv("LLM_API_TOKEN", "").strip()

JUDGE_MODEL = "gpt-4o-mini"
CONCURRENCY = 8
MAX_TOKENS_AC = 20
REQUEST_TIMEOUT = 90
MAX_RETRIES = 3

DEFAULT_INPUT = Path("data/llm_as_a_judge/uniqa/ragass_samples_uniqa_it.json")
DEFAULT_OUTPUT_DIR = Path("data/llm_as_a_judge/uniqa")
DATASETS = {"uniqa"}


def _chat_url(url: str) -> str:
    if not url:
        raise RuntimeError("LLM_API_URL non impostata nel file .env o nell'ambiente.")
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def _auth_header(token: str) -> str:
    if not token:
        raise RuntimeError("LLM_API_TOKEN non impostata nel file .env o nell'ambiente.")
    token = token.strip()
    if token.lower().startswith("bearer "):
        return token
    return f"Bearer {token}"


_SYS_CORRECTNESS_IT = (
    "Sei un valutatore rigoroso di correttezza fattuale per domande e risposte in italiano. "
    "Devi rispondere SOLO con una parola: correct oppure incorrect."
)


def _strip_think(content: str) -> str:
    content = (content or "").strip()
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


def _post_chat(system: str, user: str, max_tokens: int, temperature: float = 0.0) -> str:
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
    headers = {
        "Authorization": _auth_header(API_TOKEN),
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                _chat_url(API_URL),
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code in {429, 500, 502, 503, 504} and attempt < MAX_RETRIES:
                time.sleep(1.5 * attempt)
                continue
            r.raise_for_status()
            data = r.json()
            return _strip_think(data["choices"][0]["message"].get("content", ""))
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * attempt)
                continue
    raise last_error


async def llm_call(system: str, user: str, max_tokens: int, temperature: float = 0.0) -> str:
    return await asyncio.to_thread(_post_chat, system, user, max_tokens, temperature)


def parse_correctness(raw: str) -> float:
    if not raw or not raw.strip():
        return 0.0
    first_word = raw.lower().strip().split()[0]
    first_word = re.sub(r"[^a-z]", "", first_word)
    return float(first_word == "correct")


def normalize_answers(answers):
    if answers is None:
        return []
    if isinstance(answers, list):
        return [str(a) for a in answers]
    return [str(answers)]


async def compute_answer_correctness(question: str, answers, generated: str) -> dict:
    gold = " | ".join(normalize_answers(answers))
    gold_trunc = gold[:1200]
    gen_trunc = str(generated or "")[:1200]

    user = (
        f"Domanda: {question}\n"
        f"Risposte corrette/gold answer, una qualunque è accettabile: {gold_trunc}\n"
        f"Risposta del modello: {gen_trunc}\n\n"
        "La risposta del modello è corretta?\n"
        "Considera corretta la risposta se contiene la stessa informazione fattuale di almeno una risposta gold, "
        "anche con parole diverse. Ignora differenze minori di formulazione, articoli, maiuscole/minuscole o punteggiatura. "
        "Considera incorrect se la risposta è vuota, evasiva, non pertinente, troppo generica o contraddice la risposta gold.\n"
        "Rispondi esattamente con una sola parola: correct oppure incorrect."
    )

    raw = await llm_call(_SYS_CORRECTNESS_IT, user, MAX_TOKENS_AC, 0.0)
    return {
        "answer_correctness": parse_correctness(raw),
        "_ac_raw": raw,
    }


sem = None


async def evaluate_record(rec: dict) -> dict:
    async with sem:
        try:
            ac = await compute_answer_correctness(
                question=rec["question"],
                answers=rec.get("answers", []),
                generated=rec.get("generated", ""),
            )
            error = None
        except Exception as e:
            ac = {
                "answer_correctness": None,
                "_ac_raw": f"error: {type(e).__name__}: {str(e)[:200]}",
            }
            error = f"{type(e).__name__}: {str(e)[:200]}"

        return {
            "example_id": rec.get("example_id"),
            "dataset": rec.get("dataset"),
            "model_id": rec.get("model_id"),
            "answer_correctness": ac["answer_correctness"],
            "_ac_raw": ac["_ac_raw"],
            "error": error,
        }


def record_key(rec: dict) -> tuple:
    return (rec.get("dataset"), rec.get("model_id"), rec.get("example_id"))


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
    print(f"Judge:       {JUDGE_MODEL}")
    print(f"Record:      {len(records)} nuovi")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"API URL:     {_chat_url(API_URL)}")

    tasks = [evaluate_record(rec) for rec in records]
    details = list(existing_details)
    start_done = len(details)
    total = start_done + len(records)

    t0 = time.time()

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
    parser.add_argument("--datasets", type=str, nargs="+", default=["uniqa"])
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

    wanted_datasets = set(args.datasets)
    records = [r for r in records if r.get("dataset") in wanted_datasets]

    unknown = wanted_datasets - DATASETS
    if unknown:
        print(f"[WARN] Dataset non standard richiesti: {sorted(unknown)}")

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
