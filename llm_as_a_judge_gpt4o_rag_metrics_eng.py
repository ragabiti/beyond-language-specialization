"""
llm_as_a_judge_gpt4o_rag_metrics_eng.py

LLM-as-a-judge evaluation for English RAG samples:
- faithfulness
- context relevance
- answer relevance with n synthetic questions + local embeddings

Answer relevance follows the RAGAS-style pipeline:
    generated answer -> n synthetic questions by LLM
    original question + synthetic questions -> local embeddings
    answer_relevance = mean cosine similarity(original_question, synthetic_questions)

Input expected: JSON list with fields:
    example_id, dataset, model_id, question, generated, context
Optional fields are copied when present:
    answers, is_correct_string

Outputs:
    <output-dir>/rag_metrics_details.json
    <output-dir>/rag_metrics_summary_by_model_dataset.json

Example:
    python llm_as_a_judge_gpt4o_rag_metrics_eng.py \
      --input-file data/llm_as_a_judge/eng/ragass_samples_eng_updated.json \
      --output-dir data/llm_as_a_judge/eng_rag_metrics \
      --judge-model gpt-4o-mini \
      --concurrency 16 \
      --datasets triviaqa nq bioasq
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

load_dotenv(dotenv_path=".env", override=True)

API_URL = os.getenv("LLM_API_URL", "").strip()
API_TOKEN = os.getenv("LLM_API_TOKEN", "").strip()

JUDGE_MODEL = "gpt-4o-mini"
CONCURRENCY = 4
N_QUESTIONS = 5
CTX_MAX_CHARS = 25000
EMBEDDING_MODEL = "BAAI/bge-m3"

MAX_TOKENS_FAITH = 300
MAX_TOKENS_AR_QGEN = 80
MAX_TOKENS_CR = 400

DEFAULT_INPUT = Path("data/llm_as_a_judge/eng/ragass_samples_eng_updated.json")
DEFAULT_OUTPUT_DIR = Path("data/llm_as_a_judge/eng_rag_metrics")
DATASETS = {"triviaqa", "nq", "bioasq"}

sem = None
embedder = None
embed_lock = None
question_embedding_cache = {}


def _normalize_openai_base_url(url: str) -> str:
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


def count_sentences(text: str) -> int:
    sentences = re.split(r"[.!?]+", text or "")
    return max(1, len([s for s in sentences if s.strip()]))


# ─────────────────────────────────────────────────────────────────────────────
# Faithfulness
# ─────────────────────────────────────────────────────────────────────────────
_SYS_FAITH_EXTRACT = (
    "You extract factual statements from answers. "
    "List each statement on a new line starting with 'statement: '."
)

_SYS_FAITH_VERIFY = (
    "You verify whether statements are supported by a context. "
    "For each statement give a brief explanation, then a verdict. "
    "End with verdicts in this exact format, one per line:\n"
    "verdict: Yes\nverdict: No"
)


def parse_statements(text: str) -> list[str]:
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
        "Given a question and an answer, create one or more atomic factual statements "
        "from the answer. Ignore purely stylistic or hedging text.\n"
        f"Question: {question}\n"
        f"Answer: {gen_trunc}\n\n"
        "List each statement starting with 'statement: '."
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
        "Consider the context and determine whether each statement is supported by it. "
        "A statement is supported only if the context contains enough information to infer it. "
        "Use meaning, not exact wording.\n\n"
        f"Context:\n{context_trunc}\n\n"
        f"Statements:\n{stmts_str}\n\n"
        "End with final verdicts, one per statement, in order:\n"
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


# ─────────────────────────────────────────────────────────────────────────────
# Context relevance
# ─────────────────────────────────────────────────────────────────────────────
_SYS_CTXREL = (
    "You extract relevant sentences from a context. "
    "Output ONLY copied sentences verbatim, or 'Insufficient Information' if none are relevant."
)


async def compute_context_relevance(client: AsyncOpenAI, question: str, context: str) -> dict:
    context = str(context or "")
    if not context.strip():
        return {"context_relevance": 0.0, "n_extracted": 0, "n_total_sentences": 0, "_cr_raw": ""}

    context_trunc = context[:CTX_MAX_CHARS]
    user = (
        "From the context below, copy only the sentences that help answer the question. "
        "If none are useful, write exactly 'Insufficient Information'. "
        "Do not rewrite sentences.\n\n"
        f"Question: {question}\n"
        f"Context:\n{context_trunc}\n\n"
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


# ─────────────────────────────────────────────────────────────────────────────
# Answer relevance: n synthetic questions + local embedding cosine similarity
# ─────────────────────────────────────────────────────────────────────────────
_SYS_QGEN = (
    "You generate one English question that could be answered by the given answer. "
    "Output ONLY the question, nothing else."
)


async def generate_one_question(client: AsyncOpenAI, generated: str) -> str:
    gen_trunc = str(generated or "")[:800]
    user = (
        "Generate one plausible English question whose answer would be the text below. "
        "Keep the question specific. Output only the question.\n\n"
        f"Answer:\n{gen_trunc}"
    )
    return await llm_call(client, _SYS_QGEN, user, MAX_TOKENS_AR_QGEN, temperature=0.7)


def _clean_synthetic_question(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"^[\d]+[.)]\s*", "", q).strip()
    q = q.strip('"').strip("'").strip()
    return q


async def embed_texts(texts: list[str]) -> np.ndarray:
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
    key = question.strip()
    cached = question_embedding_cache.get(key)
    if cached is not None:
        return cached
    emb = (await embed_texts([key]))[0]
    question_embedding_cache[key] = emb
    return emb


async def compute_answer_relevance(client: AsyncOpenAI, question: str, generated: str) -> dict:
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
    return (rec.get("dataset"), rec.get("model_id"), rec.get("example_id"))


def is_valid_result(rec: dict) -> bool:
    if rec.get("error"):
        return False
    required = ["faithfulness", "context_relevance", "answer_relevance"]
    return all(rec.get(k) is not None for k in required)


async def evaluate_record(client: AsyncOpenAI, rec: dict) -> dict:
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
    global JUDGE_MODEL, CONCURRENCY, N_QUESTIONS, CTX_MAX_CHARS, EMBEDDING_MODEL
    global sem, embedder, embed_lock

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

    JUDGE_MODEL = args.judge_model
    CONCURRENCY = args.concurrency
    N_QUESTIONS = args.n_questions
    CTX_MAX_CHARS = args.ctx_max_chars
    EMBEDDING_MODEL = args.embedding_model
    sem = asyncio.Semaphore(CONCURRENCY)
    embed_lock = asyncio.Lock()

    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers non installato. Installa con: pip install sentence-transformers")

    print(f"Caricamento embedding model locale: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    details_path = output_dir / "rag_metrics_details.json"
    summary_path = output_dir / "rag_metrics_summary_by_model_dataset.json"

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

    existing, pending = load_existing_details(details_path, records)
    details = asyncio.run(run(pending, details_path, existing))

    summary = build_summary(details)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Salvato dettaglio: {details_path}")
    print(f"Salvato summary:   {summary_path}")


if __name__ == "__main__":
    main()
