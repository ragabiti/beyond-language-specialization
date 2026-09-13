"""
llm_as_a_judge_gpt4o_eng.py

Valutazione LLM-as-a-judge SOLO per Answer Correctness sui dataset ENG:
- triviaqa
- nq
- bioasq

Input atteso:
    data/ragass/ragass_samples_eng_updated.json

Campi richiesti per ogni record:
    example_id, dataset, model_id, question, answers, generated

Output:
    <output-dir>/answer_correctness_accuracy_by_model_dataset.json

Formato output:
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

Usa endpoint OpenAI-compatible da .env:
    LLM_API_URL
    LLM_API_TOKEN

Esempi:
    python llm_as_a_judge_gpt4o_eng.py \
      --input-file data/ragass/ragass_samples_eng_updated.json \
      --output-dir data/ragass/results_gpt4o_eng_ac

    python llm_as_a_judge_gpt4o_eng.py \
      --input-file data/ragass/ragass_samples_eng_updated.json \
      --output-dir data/ragass/results_gpt4o_eng_ac \
      --n-samples 50

    python llm_as_a_judge_gpt4o_eng.py \
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

load_dotenv(dotenv_path=".env", override=True)

API_URL = os.getenv("LLM_API_URL", "").strip()
API_TOKEN = os.getenv("LLM_API_TOKEN", "").strip()

JUDGE_MODEL = "gpt-4o"
CONCURRENCY = 4
MAX_TOKENS_AC = 20

DEFAULT_INPUT = Path("data/ragass/ragass_samples_eng_updated.json")
DEFAULT_OUTPUT_DIR = Path("data/ragass/results_gpt4o_eng_ac")
DATASETS = {"triviaqa", "nq", "bioasq"}


def _normalize_openai_base_url(url: str) -> str:
    if not url:
        raise RuntimeError("LLM_API_URL non impostata nel file .env o nell'ambiente.")

    url = url.rstrip("/")
    suffixes = ["/chat/completions", "/v1/chat/completions"]
    for suffix in suffixes:
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break

    if not url.endswith("/v1"):
        url = url + "/v1"

    return url


def _clean_api_key(token: str) -> str:
    if not token:
        raise RuntimeError("LLM_API_TOKEN non impostata nel file .env o nell'ambiente.")
    token = token.strip()
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()
    return token


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=_clean_api_key(API_TOKEN),
        base_url=_normalize_openai_base_url(API_URL),
    )


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


def parse_correctness(raw: str) -> float:
    if not raw.strip():
        return 0.0
    first_word = raw.lower().strip().split()[0]
    first_word = re.sub(r"[^a-z]", "", first_word)
    return float(first_word == "correct")


async def compute_answer_correctness(
    client: AsyncOpenAI,
    question: str,
    answers: list,
    generated: str,
) -> dict:
    gold = " | ".join(str(a) for a in answers)

    gold_trunc = gold[:800]
    gen_trunc = str(generated)[:800]

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

    raw = await llm_call(client, _SYS_CORRECTNESS, user, MAX_TOKENS_AC)
    return {
        "answer_correctness": parse_correctness(raw),
        "_ac_raw": raw,
    }


sem = None


async def evaluate_record(client: AsyncOpenAI, rec: dict) -> dict:
    async with sem:
        try:
            ac = await compute_answer_correctness(
                client=client,
                question=rec["question"],
                answers=rec["answers"],
                generated=rec.get("generated", ""),
            )
            error = None
        except Exception as e:
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
    client = make_client()

    print(f"Judge:       {JUDGE_MODEL}")
    print(f"Record:      {len(records)} nuovi")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"API base:    {_normalize_openai_base_url(API_URL)}")

    tasks = [evaluate_record(client, rec) for rec in records]
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
    parser.add_argument("--datasets", type=str, nargs="+", default=["triviaqa", "nq", "bioasq"])
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
