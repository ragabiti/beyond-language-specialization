"""
run_ragass_phase1_uniqa.py  —  Fase 1 RAGAS per UniQA IT

Versione adattata da run_ragass_phase1_qwen3.py per UniQA IT.

Differenze rispetto al run EN:
  - Input:  data/ragass/ragass_samples_uniqa_it.json
  - Output: data/ragass/interim_llm_results_uniqa_it.json
  - Contesto troncato a 1200 caratteri (vs 2500 EN) per rispettare
    il limite di 4096 token di Qwen3-8B con passaggi universitari lunghi
  - Try/except per record: un errore non blocca il run completo
  - Salvataggio progressivo ogni 500 record

Prerequisito (stesso del run EN):
    vllm serve Qwen/Qwen3-8B --port 8000 --dtype bfloat16 \
        --gpu-memory-utilization 0.90 --max-model-len 4096 \
        --max-num-seqs 64

Uso:
    python run_ragass_phase1_uniqa.py                  # run completo
    python run_ragass_phase1_uniqa.py --n-samples 10  # test rapido
    python run_ragass_phase1_uniqa.py --n-samples 50  # benchmark
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

# ─── Config ───────────────────────────────────────────────────────────────────
VLLM_URL    = "http://localhost:8000/v1"
JUDGE_MODEL = "Qwen/Qwen3-8B"

MAX_TOKENS_AC    = 20
MAX_TOKENS_FAITH = 300
MAX_TOKENS_AR    = 80
MAX_TOKENS_CR    = 400

N_QUESTIONS  = 5
CONCURRENCY  = 16

# Nessun troncamento — Qwen3-8B ha max_position_embeddings=40960.
# Con vLLM a --max-model-len 32768 copriamo tutto il contesto UniQA (max ~25000 token).
CTX_MAX_CHARS = 25000  # nessun crop reale

IN_PATH  = Path("data/ragass/ragass_samples_uniqa_it.json")
OUT_PATH = Path("data/ragass/interim_llm_results_uniqa_it.json")

QWEN3_EXTRA = {"enable_thinking": False}

DATASET_NAME = "uniqa_it"


# ─── Client ───────────────────────────────────────────────────────────────────
def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key="dummy", base_url=VLLM_URL)


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
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=QWEN3_EXTRA,
    )
    content = resp.choices[0].message.content.strip()
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 1: Answer Correctness
# ─────────────────────────────────────────────────────────────────────────────
_SYS_CORRECTNESS = (
    "/no_think\n"
    "You are a factual answer evaluator. "
    "Output ONLY one word: correct or incorrect."
)

async def compute_answer_correctness(
    client, question: str, answers: list, generated: str
) -> dict:
    gold = " | ".join(str(a) for a in answers)
    # Tronca gold e generated per evitare overflow
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
    first_word = re.sub(r'[^a-z]', '', raw.lower().split()[0]) if raw.strip() else ""
    binary     = float(first_word == "correct")
    return {"answer_correctness": binary, "_ac_raw": raw}


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 2: Faithfulness
# ─────────────────────────────────────────────────────────────────────────────
_SYS_FAITH_EXTRACT = (
    "/no_think\n"
    "You extract factual statements from answers. "
    "List each statement on a new line starting with 'statement: '."
)

_SYS_FAITH_VERIFY = (
    "/no_think\n"
    "You verify if statements are supported by a context. "
    "For each statement give a brief explanation, then a verdict. "
    "End with verdicts in this format — one per line:\n"
    "verdict: Yes\nverdict: No"
)

def parse_statements(text: str) -> list:
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
            m = re.match(r'^[\d]+[.)]\s+(.+)$', line)
            if m:
                statements.append(m.group(1).strip())
            elif line.startswith("- ") and len(line) > 2:
                statements.append(line[2:].strip())
    return statements

def parse_verdicts(text: str, n: int) -> list:
    verdicts = []
    for line in text.strip().split("\n"):
        line = line.strip().lower()
        if line.startswith("verdict:"):
            val = line[len("verdict:"):].strip()
            verdicts.append(val.startswith("yes"))
    if len(verdicts) != n:
        yeses = len(re.findall(r'\byes\b', text.lower()))
        verdicts = [True] * min(yeses, n) + [False] * max(0, n - yeses)
    return (verdicts + [False] * n)[:n]

async def compute_faithfulness(
    client, question: str, generated: str, context: str
) -> dict:
    if not generated.strip():
        return {"faithfulness": 1.0, "n_statements": 0, "n_verified": 0}

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

    if not statements:
        return {"faithfulness": 1.0, "n_statements": 0, "n_verified": 0}

    stmts_str     = "\n".join(f"statement: {s}" for s in statements)
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

    n_verified = sum(verdicts)
    return {
        "faithfulness": round(n_verified / len(statements), 4),
        "n_statements": len(statements),
        "n_verified":   n_verified,
    }


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 3: Answer Relevance
# ─────────────────────────────────────────────────────────────────────────────
_SYS_QGEN = "/no_think\nGeneri domande in italiano. Scrivi SOLO la domanda, nient'altro."

async def generate_one_question(client, generated: str) -> str:
    gen_trunc = generated[:300] if len(generated) > 300 else generated
    user = f"Quale domanda ha come risposta: {gen_trunc}?"
    return await llm_call(client, _SYS_QGEN, user, MAX_TOKENS_AR, temperature=0.7)

async def compute_answer_relevance_llm(client, generated: str) -> dict:
    if not generated.strip():
        return {"synthetic_questions": []}
    tasks     = [generate_one_question(client, generated) for _ in range(N_QUESTIONS)]
    questions = await asyncio.gather(*tasks)
    clean = []
    for q in questions:
        q = q.strip()
        if not q or len(q) < 5 or q.startswith(("*", "-", "**")):
            continue
        q = re.sub(r'^[\d]+[.)]\s*', '', q).strip()
        clean.append(q)
    return {"synthetic_questions": clean[:N_QUESTIONS]}


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 4: Context Relevance
# ─────────────────────────────────────────────────────────────────────────────
_SYS_CTXREL = (
    "/no_think\n"
    "You extract relevant sentences from a context. "
    "Output ONLY the extracted sentences verbatim, "
    "or 'Insufficient Information' if none are relevant."
)

def count_sentences(text: str) -> int:
    sentences = re.split(r'[.!?]+', text)
    return max(1, len([s for s in sentences if s.strip()]))

async def compute_context_relevance(client, question: str, context: str) -> dict:
    if not context.strip():
        return {"context_relevance": 0.0, "n_extracted": 0, "n_total_sentences": 0}
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
    n_extracted = 0 if "insufficient information" in raw.lower() else count_sentences(raw)
    n_total     = count_sentences(context_trunc)
    score       = min(1.0, n_extracted / n_total) if n_total > 0 else 0.0
    return {
        "context_relevance":  round(score, 4),
        "n_extracted":        n_extracted,
        "n_total_sentences":  n_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Thinking mode check
# ─────────────────────────────────────────────────────────────────────────────
def check_thinking_mode(results: list) -> bool:
    thinking_keywords = ["thinking process", "<think>", "analyze the request"]
    for rec in results[:5]:
        raw = rec.get("_ac_raw", "").lower()
        if any(kw in raw for kw in thinking_keywords):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione singolo record — con try/except per robustezza
# ─────────────────────────────────────────────────────────────────────────────
sem = asyncio.Semaphore(CONCURRENCY)

async def evaluate_record(client, rec: dict) -> dict:
    async with sem:
        try:
            ac     = await compute_answer_correctness(
                client, rec["question"], rec["answers"], rec["generated"]
            )
            faith  = await compute_faithfulness(
                client, rec["question"], rec["generated"], rec["context"]
            )
            ar_llm = await compute_answer_relevance_llm(client, rec["generated"])
            cr     = await compute_context_relevance(
                client, rec["question"], rec["context"]
            )
        except Exception as e:
            print(f"\n[WARN] Record {rec.get('example_id','?')} ({rec.get('model_id','?').split('/')[-1]}) fallito: {type(e).__name__}: {str(e)[:80]}")
            ac     = {"answer_correctness": None, "_ac_raw": f"error: {type(e).__name__}"}
            faith  = {"faithfulness": None, "n_statements": 0, "n_verified": 0}
            ar_llm = {"synthetic_questions": []}
            cr     = {"context_relevance": None, "n_extracted": 0, "n_total_sentences": 0}

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
    client   = make_client()
    n_calls  = len(records) * 9
    n_models  = len(set(r["model_id"] for r in records))
    n_samples = len(set(r["example_id"] for r in records))

    print(f"\n  Dataset:      {DATASET_NAME}")
    print(f"  Record:       {len(records)}  ({n_models} modelli × {n_samples} sample)")
    print(f"  LLM calls:    {n_calls}  (9/record: 1 AC + 2 Faith + {N_QUESTIONS} AR + 1 CR)")
    print(f"  Concurrency:  {CONCURRENCY}")
    print(f"  Judge:        {JUDGE_MODEL}")
    print(f"  Context max chars: {CTX_MAX_CHARS}")
    print(f"  Thinking:     disabled via extra_body\n")

    t0      = time.time()
    tasks   = [evaluate_record(client, rec) for rec in records]
    results = []

    async for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await coro)

        # Salvataggio progressivo ogni 500 record
        if len(results) % 250 == 0:
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
            elapsed_so_far = time.time() - t0
            print(f"\n  Checkpoint: {len(results)}/{len(records)} record  "
                  f"({elapsed_so_far/60:.1f} min)")

    elapsed = time.time() - t0
    sec_per = elapsed / len(results) if results else 0

    if check_thinking_mode(results):
        print("\n⚠️  ATTENZIONE: thinking mode ancora attivo!")
    else:
        print(f"\n✓ Thinking mode: disabilitato correttamente")

    errors = sum(1 for r in results if r.get("_ac_raw", "").startswith("error:"))
    if errors:
        print(f"⚠️  Record con errori: {errors}/{len(results)}")

    print(f"✓ Completato in {elapsed/60:.1f} min  ({sec_per:.2f} sec/record)")
    return results


def main():
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

    in_path  = Path(args.input_file) if args.input_file else IN_PATH
    out_dir  = Path(args.output_dir) if args.output_dir else in_path.parent
    out_path = out_dir / "interim_llm_results.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Caricamento: {in_path}")
    with open(in_path, encoding="utf-8") as f:
        records = json.load(f)

    if args.models:
        records = [r for r in records if r["model_id"] in args.models]
        print(f"Filtro modelli: {args.models}  ({len(records)} record)")

    if args.n_samples:
        records = records[:args.n_samples]
        print(f"Test/Benchmark: {len(records)} record")
    else:
        print(f"Record totali: {len(records)}")

    results = asyncio.run(run(records, out_path))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Salvato: {out_path}  ({len(results)} record)")
    print(f"  Ora spegni vLLM e lancia: python run_ragass_phase2_uniqa.py --input-file {out_path} --output-dir {out_dir}")


if __name__ == "__main__":
    main()
