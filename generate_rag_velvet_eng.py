"""
generate_rag_velvet_eng_unified.py

Generazione risposte RAG per dataset inglesi con modelli Velvet.
Versione aggiornata con le accortezze usate negli script UniQA/PharmaQA:
  - prompt non limitato a "max 5 tokens", ma più secco e dataset-specific
  - cleaning robusto dell'output
  - salvataggio progressivo
  - accuracy solo indicativa; metrica principale: RAGAS / LLM-as-a-judge
  - max_new_tokens default basso: 120

Uso:
    python generate_rag_velvet_eng_unified.py \
        --llm_id velvet-14b \
        --datasets triviaqa nq bioasq \
        --k_values 5 \
        --output_dir data/gen_res

Dataset attesi:
    data/triviaqa_dataset.json
    data/nq_dataset.json
    data/popqa_dataset.json
    data/bioasq_dataset.json
"""

import os
import sys
import re
import json
import pickle
import argparse
from tqdm import tqdm

sys.path.insert(0, "src")
from llm_velvet_uniqa import LLM


DATASET_PATHS = {
    "triviaqa": "data/dataset/triviaqa_dataset.json",
    "nq": "data/dataset/nq_dataset.json",
    "popqa": "data/dataset/popqa_dataset.json",
    "bioasq": "data/dataset/bioasq_dataset.json",
}


TASK_INSTRUCTION_NQ = (
    "You are given a question and retrieved documents. "
    "Answer using only information explicitly supported by the documents. "
    "Give the shortest complete answer possible. "
    "Prefer a noun phrase or one short sentence. "
    "Do not explain. Do not add background information. "
    "If the answer is not present, respond exactly with NO-RES."
)

TASK_INSTRUCTION_TRIVIAQA = (
    "You are given a trivia question and retrieved documents. "
    "Answer using only information explicitly supported by the documents. "
    "Give only the answer entity or the shortest possible answer. "
    "Do not explain. Do not add background information. "
    "If the answer is not present, respond exactly with NO-RES."
)

TASK_INSTRUCTION_BIOASQ = (
    "You are given a biomedical question and retrieved biomedical documents. "
    "Answer using only information explicitly supported by the documents. "
    "Give the shortest complete biomedical answer possible. "
    "Prefer a biomedical entity, noun phrase, or one short sentence. "
    "Do not explain. Do not add background information. "
    "If the answer is not present, respond exactly with NO-RES."
)

TASK_INSTRUCTIONS = {
    "triviaqa": TASK_INSTRUCTION_TRIVIAQA,
    "nq": TASK_INSTRUCTION_NQ,
    "popqa": TASK_INSTRUCTION_TRIVIAQA,
    "bioasq": TASK_INSTRUCTION_BIOASQ,
}


LIMITED_CONTEXT_MODELS = []
PASSAGE_MAX_TOKENS = 500


def _truncate_passage(text: str, max_tokens: int = PASSAGE_MAX_TOKENS) -> str:
    """Troncamento opzionale per modelli con context window limitata."""
    max_words = int(max_tokens * 0.75)
    words = str(text).split()
    if len(words) <= max_words:
        return str(text)
    return " ".join(words[:max_words]) + "..."


def build_prompt(question: str, passages: list, dataset_name: str, llm_id: str = "") -> str:
    instruction = TASK_INSTRUCTIONS.get(dataset_name, TASK_INSTRUCTION_NQ)

    llm_lower = llm_id.lower()
    needs_truncation = any(m in llm_lower for m in LIMITED_CONTEXT_MODELS)

    docs_str = "\n".join(
        f"Document [{i+1}](Title: passage) "
        f"{_truncate_passage(p.get('text', '')) if needs_truncation else p.get('text', '')}"
        for i, p in enumerate(passages)
    )
    return f"{instruction}\nDocuments:\n{docs_str}\nQuestion: {question}\nAnswer:"


def clean_generated(output: str) -> str:
    """
    Cleaning robusto per output Velvet.
    Gestisce risposta pura, prompt ripetuto, thinking tags, token artefatti e secondo turno.
    """
    if output is None:
        return ""

    output = str(output)

    if "Answer:" in output:
        output = output[output.find("Answer:") + len("Answer:"):]

    if "</think>" in output:
        output = output[output.rfind("</think>") + len("</think>"):]

    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)

    preambles = [
        "Sure!", "Of course!", "Certainly!", "Here is the answer:", "The answer is:",
        "Based on the documents,", "Based on the provided documents,",
        "According to the documents,", "According to the provided documents,",
    ]
    stripped = output.strip()
    for p in preambles:
        if stripped.startswith(p):
            stripped = stripped[len(p):].strip()
            break
    output = stripped

    for stop in ["Question:", "Document [", "Documents:", "Human:", "User:", "Assistant:", "\nAnswer:"]:
        if stop in output:
            output = output[:output.find(stop)]

    output = output.replace("\u0120", " ")
    output = output.replace("\u010a", "\n")
    output = re.sub(r" +", " ", output)
    output = re.sub(r"\n{3,}", "\n\n", output)

    return output.strip()


def _normalize(text: str) -> str:
    text = re.sub(r"\*+", "", str(text))
    text = re.sub(r"#+\s", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def _key_phrases(text: str, min_words: int = 4) -> list[str]:
    sentences = [s.strip() for s in str(text).replace("\n", ". ").split(".") if s.strip()]
    return [_normalize(s) for s in sentences if len(s.split()) >= min_words]


def string_match(generated: str, answers: list) -> bool:
    """
    Match indicativo. Non usare come metrica principale.
    La valutazione principale resta RAGAS / LLM-as-a-judge.
    """
    gen = _normalize(generated)
    for sep in ["\n", "Document [", "Question:", "Human:", "User:"]:
        if sep.lower() in gen:
            gen = gen.split(sep.lower())[0].strip()

    for ans in answers:
        ans_norm = _normalize(ans)
        if ans_norm and ans_norm in gen:
            return True
        for phrase in _key_phrases(ans):
            if phrase and phrase in gen:
                return True
    return False


def _safe_n_relevant(passages: list) -> int:
    return sum(1 for p in passages if p.get("is_relevant", False))


def run_dataset(llm, dataset_name: str, dataset_path: str, args, llm_folder: str):
    print(f"\n{'#'*60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'#'*60}")
    print(f"Caricamento: {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples:
        data = data[:args.max_samples]

    llm_lower = args.llm_id.lower()
    if any(m in llm_lower for m in LIMITED_CONTEXT_MODELS):
        print(
            f"  [context limit] Passaggi troncati a {PASSAGE_MAX_TOKENS} token "
            f"({int(PASSAGE_MAX_TOKENS * 0.75)} parole)"
        )

    print(f"Esempi: {len(data)}")

    for k in args.k_values:
        print(f"\n{'='*50}")
        print(f"Dataset={dataset_name}  k={k} passages")
        print(f"{'='*50}")

        save_dir = os.path.join(args.output_dir, llm_folder, dataset_name, f"top{k}")
        os.makedirs(save_dir, exist_ok=True)

        summary_path = os.path.join(save_dir, "summary.json")
        if not args.overwrite and os.path.exists(summary_path):
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            print(
                f"  Già completato — Accuracy indicativa: {summary['accuracy']:.4f} "
                f"(usa --overwrite per rigirare)"
            )
            continue

        results = []
        correct = 0

        for idx, example in enumerate(tqdm(data, desc=f"{dataset_name} k={k}")):
            passages = example["passages"][:k]
            prompt = build_prompt(example["question"], passages, dataset_name, args.llm_id)

            output = llm.generate(prompt, max_new_tokens=args.max_new_tokens)
            if isinstance(output, list):
                output = output[0]

            generated = clean_generated(output)

            is_correct = string_match(generated, example.get("answers", []))
            if is_correct:
                correct += 1

            results.append({
                "example_id": example.get("example_id", idx),
                "question": example.get("question", ""),
                "answers": example.get("answers", []),
                "generated": generated,
                "is_correct": is_correct,
                "k": k,
                "dataset": dataset_name,
                "n_relevant": _safe_n_relevant(passages),
            })

            if (idx + 1) % args.save_every == 0 or (idx + 1) == len(data):
                acc = correct / (idx + 1)
                print(f"  [{idx+1}/{len(data)}] Accuracy indicativa: {acc:.4f}")
                fname = os.path.join(save_dir, f"results_top{k}_info_{idx+1}.pkl")
                with open(fname, "wb") as f:
                    pickle.dump(results, f)

        final_acc = correct / len(data) if data else 0.0
        print(f"\n{dataset_name} k={k} — Accuracy indicativa: {final_acc:.4f}")
        print("  (metrica principale: RAGAS con LLM judge)")

        summary = {
            "llm_id": args.llm_id,
            "dataset": dataset_name,
            "k": k,
            "n_samples": len(data),
            "accuracy": round(final_acc, 4),
            "accuracy_note": "indicativa — metrica principale: RAGAS / LLM-as-a-judge",
            "max_new_tokens": args.max_new_tokens,
            "prompt_style": "short grounded answer, dataset-specific instruction, no max-5-token extraction",
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_id", type=str, required=True)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["triviaqa", "nq", "bioasq"],
        help=f"Dataset da usare. Disponibili: {list(DATASET_PATHS.keys())}",
    )
    parser.add_argument("--data_dir", type=str, default="data/dataset")
    parser.add_argument("--k_values", type=int, nargs="+", default=[5])
    parser.add_argument("--output_dir", type=str, default="data/gen_res")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    for ds in args.datasets:
        if ds not in DATASET_PATHS:
            raise ValueError(f"Dataset '{ds}' non riconosciuto. Disponibili: {list(DATASET_PATHS.keys())}")

    print(f"Modello:        {args.llm_id}")
    print(f"Dataset:        {args.datasets}")
    print(f"k values:       {args.k_values}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"Output:         {args.output_dir}")

    llm = LLM(args.llm_id)
    llm_folder = args.llm_id.split("/")[-1] if "/" in args.llm_id else args.llm_id

    for dataset_name in args.datasets:
        default_filename = os.path.basename(DATASET_PATHS[dataset_name])
        dataset_path = os.path.join(args.data_dir, default_filename)

        if not os.path.exists(dataset_path):
            print(f"\nATTENZIONE: {dataset_path} non trovato, skip {dataset_name}.")
            continue

        run_dataset(llm, dataset_name, dataset_path, args, llm_folder)

    print("\nGenerazione completata!")


if __name__ == "__main__":
    main()
