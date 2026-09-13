"""
generate_rag_uniqa_vllm.py — Generazione RAG su UniQA via vLLM per tutti i modelli.

Usa src/llm_vllm_chat.py per tutti i modelli non-Velvet.
llm_vllm_chat.py gestisce automaticamente:
  - Chat template  → /v1/chat/completions  (gemma, granite, ministral, fastweb, villanova)
  - Raw completion → /v1/completions        (llama, qwen, altri)
  - Thinking mode  → soppresso per Qwen

Nessun crop — contesto passato intero a vLLM.

Prerequisito:
    vllm serve <model_id> \\
        --port 8000 --dtype bfloat16 \\
        --gpu-memory-utilization 0.92 \\
        --max-model-len 32768 \\
        --max-num-seqs 8 \\
        --disable-log-requests

Uso:
    python generate_rag_uniqa_vllm.py --llm_id google/gemma-4-E2B-it
    python generate_rag_uniqa_vllm.py --llm_id Qwen/Qwen3.5-9B
    python generate_rag_uniqa_vllm.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import os
import re
import sys
import json
import pickle
import argparse
from tqdm import tqdm


def _load_llm_class(model_id: str):
    """
    Carica la classe LLM appropriata:
      - Velvet (API) → src/llm_velvet_uniqa.py
      - tutti gli altri → src/llm_vllm_chat.py (gestisce chat e raw completion)
    """
    sys.path.insert(0, "src")
    if "velvet" in model_id.lower():
        from llm_velvet_uniqa import LLM
    else:
        from llm_vllm_chat import LLM
    return LLM


DATASET_PATHS = {
    'uniqa_it': 'data/dataset/uniqa_it_dataset.json',
    'uniqa_en': 'data/dataset/uniqa_it_dataset.json',
}

# Modelli con context window limitata a 4096 token.
# I passaggi vengono troncati per evitare overflow → generazione lenta e degenere.
# Nota metodologica: limitazione del modello, dichiarata nel paper.
LIMITED_CONTEXT_MODELS = [ ]
PASSAGE_MAX_TOKENS = 500 #350  # token per passaggio (approssimazione: ~260 parole)


def _truncate_passage(text: str, max_tokens: int = PASSAGE_MAX_TOKENS) -> str:
    """Tronca un passaggio a max_tokens (approssimazione rapida via word count)."""
    # 1 token ≈ 0.75 parole per italiano
    max_words = int(max_tokens * 0.75)
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + '...'

# ── Prompt ────────────────────────────────────────────────────────────────────
# Differenza chiave rispetto a generate_rag.py:
# - niente "max 5 tokens"
# - richiede risposta completa e articolata
# - specifica italiano per uniqa_it

TASK_INSTRUCTION_IT = (
    "Sei un assistente universitario. Ti viene fornita una domanda e una serie di documenti. "
    "Rispondi alla domanda IN ITALIANO in modo completo e preciso, "
    "basandoti ESCLUSIVAMENTE sulle informazioni presenti nei documenti forniti. "
    "Se i documenti non contengono informazioni sufficienti, rispondi con NO-RES."
)

TASK_INSTRUCTION_EN = (
    "You are a university assistant. You are given a question and a series of documents. "
    "Answer the question in a complete and accurate way, "
    "based EXCLUSIVELY on the information in the provided documents. "
    "If the documents do not contain sufficient information, respond with NO-RES."
)

TASK_INSTRUCTIONS = {
    'uniqa_it': TASK_INSTRUCTION_IT,
    'uniqa_en': TASK_INSTRUCTION_EN,
}


def build_prompt(question: str, passages: list, dataset_name: str, llm_id: str = "") -> str:
    instruction = TASK_INSTRUCTIONS.get(dataset_name, TASK_INSTRUCTION_IT)

    # Tronca passaggi per modelli con context window limitata
    llm_lower = llm_id.lower()
    needs_truncation = any(m in llm_lower for m in LIMITED_CONTEXT_MODELS)

    docs_str = "\n".join(
        f"Document [{i+1}](Title: passage) {_truncate_passage(p['text']) if needs_truncation else p['text']}"
        for i, p in enumerate(passages)
    )
    return f"{instruction}\nDocuments:\n{docs_str}\nQuestion: {question}\nAnswer:"


# ── Pulizia output modello ────────────────────────────────────────────────────

def clean_generated(output: str) -> str:
    """
    Pulisce l'output grezzo del modello in modo robusto per tutti i modelli.
    Gestisce:
      - Qwen3.5 / Qwen3-14B : <think>...</think> prima della risposta
      - Llama, Gemma, altri  : preamble tipo "Certo!", "Sure!", ecc.
      - Tutti                : ripetizione del prompt, secondo turno
    """
    # 1. Estrai dopo "Answer:" se il modello lo ripete
    if "Answer:" in output:
        output = output[output.find("Answer:") + len("Answer:"):]

    # 2. Rimuovi thinking tags (Qwen3.5, Qwen3-14B thinking mode)
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)

    # 3. Rimuovi preamble comuni (Gemma, alcuni Llama)
    preambles = [
        "Certo!", "Certamente!", "Ecco la risposta:", "Ecco:",
        "Sure!", "Of course!", "Certainly!",
        "Based on the documents,", "Based on the provided documents,",
        "In base ai documenti forniti,",
    ]
    stripped = output.strip()
    for p in preambles:
        if stripped.startswith(p):
            stripped = stripped[len(p):]
            break
    output = stripped

    # 4. Taglia se il modello inizia un secondo turno
    for stop in ["Question:", "Document [1]", "Human:", "User:", "\nAnswer:"]:
        if stop in output:
            output = output[:output.find(stop)]

    return output.strip()


# ── String match ──────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Normalizza testo: rimuove markdown, spazi multipli, newline."""
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def _key_phrases(text: str, min_words: int = 5) -> list[str]:
    """Estrae frasi di almeno min_words parole dal testo."""
    sentences = [s.strip() for s in text.replace('\n', '. ').split('.') if s.strip()]
    return [_normalize(s) for s in sentences if len(s.split()) >= min_words]


def string_match(generated: str, answers: list) -> bool:
    """
    Match unidirezionale su frasi chiave.
    NOTA: metrica secondaria su UniQA — la metrica principale è RAGAS.
    """
    gen = _normalize(generated)

    for ans in answers:
        # 1. Match diretto
        if _normalize(ans) in gen:
            return True
        # 2. Match su frasi chiave (≥5 parole)
        for phrase in _key_phrases(ans):
            if phrase in gen:
                return True

    return False


# ── Run per dataset ───────────────────────────────────────────────────────────

def run_dataset(llm, dataset_name: str, dataset_path: str, args, llm_folder: str):
    print(f"\n{'#'*60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'#'*60}")

    print(f"Caricamento: {dataset_path}")
    with open(dataset_path, encoding='utf-8') as f:
        data = json.load(f)

    if args.max_samples:
        data = data[:args.max_samples]

    llm_lower = args.llm_id.lower()
    if any(m in llm_lower for m in LIMITED_CONTEXT_MODELS):
        print(f"  [context limit] Passaggi troncati a {PASSAGE_MAX_TOKENS} token ({int(PASSAGE_MAX_TOKENS*0.75)} parole) — limitazione del modello")

    print(f"Esempi: {len(data)}")

    for k in args.k_values:
        print(f"\n{'='*50}")
        print(f"Dataset={dataset_name}  k={k} passages")
        print(f"{'='*50}")

        save_dir = os.path.join(args.output_dir, llm_folder, dataset_name, f"top{k}")
        os.makedirs(save_dir, exist_ok=True)

        summary_path = os.path.join(save_dir, 'summary.json')
        if not args.overwrite and os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"  Già completato — Accuracy: {summary['accuracy']:.4f}  (usa --overwrite per rigirare)")
            continue

        results = []
        correct = 0

        for idx, example in enumerate(tqdm(data, desc=f"{dataset_name} k={k}")):
            passages = example['passages'][:k]
            prompt   = build_prompt(example['question'], passages, dataset_name, args.llm_id)

            output = llm.generate(prompt, max_new_tokens=args.max_new_tokens)
            if isinstance(output, list):
                output = output[0]

            # Pulizia robusta output
            generated = clean_generated(output)

            is_correct = string_match(generated, example['answers'])
            if is_correct:
                correct += 1

            results.append({
                'example_id': example['example_id'],
                'question':   example['question'],
                'answers':    example['answers'],
                'generated':  generated,
                'is_correct': is_correct,
                'k':          k,
                'dataset':    dataset_name,
                'n_relevant': sum(1 for p in passages if p['is_relevant']),
            })

            # Salva progressivamente
            if (idx + 1) % args.save_every == 0 or (idx + 1) == len(data):
                acc = correct / (idx + 1)
                print(f"  Salvato a {idx+1} — Accuracy (indicativa): {acc:.4f}")
                fname = os.path.join(save_dir, f"results_top{k}_info_{idx+1}.pkl")
                with open(fname, 'wb') as f:
                    pickle.dump(results, f)

        final_acc = correct / len(data)
        print(f"\n{dataset_name} k={k} — Accuracy indicativa: {final_acc:.4f}")
        print(f"  (metrica principale: RAGAS con LLM judge)")

        summary = {
            'llm_id':        args.llm_id,
            'dataset':       dataset_name,
            'k':             k,
            'n_samples':     len(data),
            'accuracy':      round(final_acc, 4),
            'accuracy_note': 'indicativa — metrica principale e RAGAS',
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


# ── Argomenti ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_id',         type=str, required=True)
    parser.add_argument('--datasets',       type=str, nargs='+',
                        default=['uniqa_it'],
                        help=f"Dataset da usare. Disponibili: {list(DATASET_PATHS.keys())}")
    parser.add_argument('--data_dir',       type=str, default='data')
    parser.add_argument('--k_values',       type=int, nargs='+', default=[5])
    parser.add_argument('--output_dir',     type=str, default='data/gen_res')
    parser.add_argument('--save_every',     type=int, default=100)
    parser.add_argument('--max_samples',    type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=300,
                        help="Token massimi generati per risposta (default: 300)")
    parser.add_argument('--overwrite',      action='store_true')
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

    LLM = _load_llm_class(args.llm_id)
    llm = LLM(args.llm_id)
    llm_folder = args.llm_id.split("/")[1] if '/' in args.llm_id else args.llm_id

    for dataset_name in args.datasets:
        dataset_path = DATASET_PATHS[dataset_name]
        if not os.path.exists(dataset_path):
            print(f"\nATTENZIONE: {dataset_path} non trovato, skip {dataset_name}.")
            continue
        run_dataset(llm, dataset_name, dataset_path, args, llm_folder)

    print("\nGenerazione completata!")


if __name__ == "__main__":
    main()
