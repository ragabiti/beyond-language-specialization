"""
run_ragass_phase1_bioasq_ctx12000.py  —  Fase 1: RAGAS solo BioASQ, tutti i modelli

Judge: Qwen/Qwen3-8B  (thinking mode disabilitato via extra_body)

Metriche (reimplementazione Es et al. 2023):
  1. Answer Correctness  — 1 LLM call  [metrica custom, non in Es et al. 2023]
  2. Faithfulness        — 2 LLM calls
  3. Answer Relevance    — 5 LLM calls (1 per domanda, temp=0.7)
  4. Context Relevance   — 1 LLM call

Totale: 9 LLM calls per record

Prerequisito:
    vllm serve Qwen/Qwen3-8B --port 8000 --dtype bfloat16 \\
        --gpu-memory-utilization 0.90 --max-model-len 8192 \\
        --max-num-seqs 64

Uso:
    python run_ragass_phase1_bioasq_ctx12000.py \
      --input-file data/ragass/ragass_samples_eng_updated.json \
      --output-dir data/ragass/results_bioasq_ctx12000

    python run_ragass_phase1_bioasq_ctx12000.py \
      --input-file data/ragass/ragass_samples_eng_updated.json \
      --output-dir data/ragass/results_bioasq_ctx12000 \
      --n-samples 50

    python run_ragass_phase1_bioasq_ctx12000.py \
      --input-file data/ragass/ragass_samples_eng_updated.json \
      --output-dir data/ragass/results_bioasq_ctx12000 \
      --dataset bioasq \
      --ctx-max-chars 12000
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
TARGET_DATASET = "bioasq"

MAX_TOKENS_AC    = 20
MAX_TOKENS_FAITH = 300
MAX_TOKENS_AR    = 80
MAX_TOKENS_CR    = 400

N_QUESTIONS = 5
CONCURRENCY = 64
IN_PATH     = Path("data/ragass/ragass_samples_eng.json")

# Limite caratteri di default per il contesto nelle chiamate Faithfulness e Context Relevance.
# Può essere sovrascritto da CLI con --ctx-max-chars.
CTX_MAX_CHARS = 12000

# Disabilita thinking mode per Qwen3
# Se non funziona (output contiene ancora "Thinking Process:" o "<think>"),
# lo script lo rileva automaticamente e avvisa.
QWEN3_EXTRA = {"enable_thinking": False}


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

    # Fallback: se thinking mode ancora attivo, estrai dopo </think>
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
    gold = " | ".join(answers)
    user = (
        f"Question: {question}\n"
        f"Gold answers (any one is acceptable): {gold}\n"
        f"Model answer: {generated}\n\n"
        "Is the model answer correct? An answer is correct if it contains "
        "the same factual information as any gold answer, even if phrased differently.\n"
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

    user_extract = (
        "Given a question and answer, create one or more statements "
        "from each sentence in the given answer.\n"
        f"question: {question}\n"
        f"answer: {generated}\n\n"
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
_SYS_QGEN = "/no_think\nYou generate questions. Output ONLY the question, nothing else."

async def generate_one_question(client, generated: str) -> str:
    user = f"What question has this answer: {generated}?"
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
# Thinking mode check — avvisa se Qwen3 genera ancora thinking
# ─────────────────────────────────────────────────────────────────────────────
def check_thinking_mode(results: list) -> bool:
    """Controlla se il thinking mode è ancora attivo nei primi record."""
    thinking_keywords = ["thinking process", "<think>", "analyze the request"]
    for rec in results[:5]:
        raw = rec.get("_ac_raw", "").lower()
        if any(kw in raw for kw in thinking_keywords):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione singolo record
# ─────────────────────────────────────────────────────────────────────────────
sem = asyncio.Semaphore(CONCURRENCY)

async def evaluate_record(client, rec: dict) -> dict:
    async with sem:
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
    return {
        "example_id":        rec["example_id"],
        "dataset":           rec["dataset"],
        "model_id":          rec["model_id"],
        "question":          rec["question"],
        "generated":         rec["generated"],
        "answers":           rec["answers"],
        "is_correct_string": rec["is_correct_string"],
        **ac, **faith, **ar_llm, **cr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
async def run(records: list) -> list:
    client  = make_client()
    n_calls = len(records) * 9
    print(f"\n  Record:      {len(records)}")
    print(f"  LLM calls:   {n_calls}  (9/record: 1 AC + 2 Faith + {N_QUESTIONS} AR + 1 CR)")
    print(f"  Concurrency: {CONCURRENCY}")
    print(f"  Judge:       {JUDGE_MODEL}")
    print(f"  Thinking:    disabled via extra_body\n")

    t0      = time.time()
    tasks   = [evaluate_record(client, rec) for rec in records]
    results = []
    async for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await coro)

    elapsed = time.time() - t0
    sec_per = elapsed / len(results)

    # Controlla thinking mode
    if check_thinking_mode(results):
        print("\n⚠️  ATTENZIONE: thinking mode ancora attivo!")
        print("   Le risposte AC contengono testo del reasoning.")
        print("   Interrompi e verifica il flag --override-generation-config.")
    else:
        print(f"\n✓ Thinking mode: disabilitato correttamente")

    print(f"✓ Completato in {elapsed/60:.1f} min  ({sec_per:.2f} sec/record)")
    print(f"  Stima run completo sui record filtrati: {sec_per*len(records)/3600:.1f} ore")

    return results


def main():
    global CTX_MAX_CHARS

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=None,
                        help="N record (es: 10 per test, 50 per benchmark)")
    parser.add_argument("--output-dir", type=str, default="data/ragass",
                        help="Cartella di output (default: data/ragass)")
    parser.add_argument("--input-file", type=str, default=None,
                        help="File JSON di input (default: data/ragass/ragass_samples_eng_updated.json)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Opzionale: lista di model_id da valutare. Default: tutti i modelli presenti in BioASQ")
    parser.add_argument("--dataset", type=str, default=TARGET_DATASET,
                        help=f"Dataset da valutare. Default: {TARGET_DATASET}")
    parser.add_argument("--ctx-max-chars", type=int, default=CTX_MAX_CHARS,
                        help=f"Massimo numero di caratteri del contesto usati per Faithfulness e Context Relevance. Default: {CTX_MAX_CHARS}")
    args = parser.parse_args()

    CTX_MAX_CHARS = args.ctx_max_chars

    in_path  = Path(args.input_file) if args.input_file else IN_PATH
    out_dir  = Path(args.output_dir)
    out_path = out_dir / f"interim_llm_results_{args.dataset}_all_models_ctx{CTX_MAX_CHARS}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_path, encoding="utf-8") as f:
        records = json.load(f)

    records = [r for r in records if r.get("dataset") == args.dataset]
    print(f"Filtro dataset: {args.dataset}  ({len(records)} record)")
    if not records:
        print("⚠️  Nessun record trovato. Controlla il nome del dataset nel file di input.")
        return

    if args.models:
        records = [r for r in records if r.get("model_id") in args.models]
        print(f"Filtro modelli: {args.models}  ({len(records)} record)")
        if not records:
            print("⚠️  Nessun record trovato. Controlla i model_id nel file di input.")
            return
    else:
        models = sorted({r.get("model_id") for r in records})
        print(f"Modelli inclusi: {len(models)}")
        for m in models:
            print(f"  - {m}")

    if args.n_samples:
        records = records[:args.n_samples]
        print(f"Test/Benchmark: {len(records)} record")
    else:
        print(f"Record totali: {len(records)}")

    print(f"CTX_MAX_CHARS: {CTX_MAX_CHARS}")

    results = asyncio.run(run(records))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Risultati salvati: {out_path}")
    print(f"  Ora spegni vLLM e lancia la phase2 usando questo JSON come input.")


if __name__ == "__main__":
    main()
