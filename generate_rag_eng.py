"""
Generazione risposte RAG con modelli Velvet su più dataset.
Simula RAG reale usando i top-k passage già recuperati.

Uso (singolo dataset):
    python generate_rag.py \
        --llm_id velvet-25b-011-p10 \
        --datasets triviaqa \
        --k_values 1 2 3 5 10 15 20 25

Uso (tutti i dataset):
    python generate_rag.py \
        --llm_id velvet-25b-011-p10 \
        --datasets triviaqa nq popqa \
        --k_values 1 2 3 5 10 15 20 25

Struttura output:
    data/gen_res/
    └── velvet-25b-011-p10/
        ├── triviaqa/
        │   ├── top1/
        │   │   ├── results_top1_info_100.pkl
        │   │   └── summary.json
        │   └── top5/
        ├── nq/
        └── popqa/
"""

import os
import sys
import json
import pickle
import argparse
from tqdm import tqdm
sys.path.insert(0, "src")
from llm import LLM


# Mappa nome dataset -> path di default
DATASET_PATHS = {
    'triviaqa': 'data/triviaqa_dataset.json',
    'nq':       'data/nq_dataset.json',
    'popqa':    'data/popqa_dataset.json',
    'bioasq':   'data/bioasq_dataset.json',
}

TASK_INSTRUCTION = (
    "You are given a question and you MUST respond by EXTRACTING the answer "
    "(max 5 tokens) from one of the provided documents. "
    "If none of the documents contain the answer, respond with NO-RES."
)


def build_prompt(question: str, passages: list) -> str:
    docs_str = "\n".join(
        f"Document [{i+1}](Title: passage) {p['text']}"
        for i, p in enumerate(passages)
    )
    return f"{TASK_INSTRUCTION}\nDocuments:\n{docs_str}\nQuestion: {question}\nAnswer:"


def string_match(generated: str, answers: list) -> bool:
    gen = generated.strip().lower()
    for sep in ['\n', 'Document [', '(']:
        if sep in gen:
            gen = gen.split(sep)[0]
    gen = gen.strip()
    return any(ans.lower() in gen for ans in answers)


def run_dataset(llm, dataset_name: str, dataset_path: str, args, llm_folder: str):
    print(f"\n{'#'*60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'#'*60}")

    print(f"Caricamento: {dataset_path}")
    with open(dataset_path) as f:
        data = json.load(f)

    if args.max_samples:
        data = data[:args.max_samples]

    print(f"Esempi: {len(data)}")

    for k in args.k_values:
        print(f"\n{'='*50}")
        print(f"Dataset={dataset_name}  k={k} passage")
        print(f"{'='*50}")

        save_dir = os.path.join(args.output_dir, llm_folder, dataset_name, f"top{k}")
        os.makedirs(save_dir, exist_ok=True)

        # Skip se già completato
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
            prompt   = build_prompt(example['question'], passages)

            output = llm.generate(prompt, max_new_tokens=args.max_new_tokens)
            if isinstance(output, list):
                output = output[0]

            if "Answer:" in output:
                start = output.find("Answer:") + len("Answer:")
                generated = output[start:].strip()
            else:
                generated = output.strip()

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
                print(f"  Salvato a {idx+1} — Accuracy: {acc:.4f}")
                fname = os.path.join(save_dir, f"results_top{k}_info_{idx+1}.pkl")
                with open(fname, 'wb') as f:
                    pickle.dump(results, f)

        final_acc = correct / len(data)
        print(f"\n{dataset_name} k={k} — Accuracy finale: {final_acc:.4f}")

        summary = {
            'llm_id':    args.llm_id,
            'dataset':   dataset_name,
            'k':         k,
            'n_samples': len(data),
            'accuracy':  round(final_acc, 4),
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_id',         type=str, required=True)
    parser.add_argument('--datasets',       type=str, nargs='+', default=list(DATASET_PATHS.keys()),
                        help=f"Dataset da usare. Scegli tra: {list(DATASET_PATHS.keys())}. Default: tutti.")
    parser.add_argument('--data_dir',       type=str, default='data/dataset',
                        help="Cartella base dove cercare i dataset (default: data/)")
    parser.add_argument('--k_values',       type=int, nargs='+', default=[1, 2, 3, 5, 10, 15, 20, 25])
    parser.add_argument('--output_dir',     type=str, default='data/gen_res')
    parser.add_argument('--save_every',     type=int, default=100)
    parser.add_argument('--max_samples',    type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=15,
                        help="Token massimi generati per risposta (default: 15, aumentare a 50 per modelli verbosi)")
    parser.add_argument('--overwrite',      action='store_true',
                        help="Riesegui anche i k già completati (default: skip)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Valida dataset richiesti
    for ds in args.datasets:
        if ds not in DATASET_PATHS:
            raise ValueError(f"Dataset '{ds}' non riconosciuto. Disponibili: {list(DATASET_PATHS.keys())}")

    print(f"Modello:   {args.llm_id}")
    print(f"Dataset:   {args.datasets}")
    print(f"k values:  {args.k_values}")
    print(f"Output:    {args.output_dir}")

    llm = LLM(args.llm_id)
    llm_folder = args.llm_id.split("/")[1] if '/' in args.llm_id else args.llm_id

    for dataset_name in args.datasets:
        # Costruisce il path del dataset rispetto a data_dir
        default_filename = os.path.basename(DATASET_PATHS[dataset_name])
        dataset_path = os.path.join(args.data_dir, default_filename)

        if not os.path.exists(dataset_path):
            print(f"\nATTENZIONE: {dataset_path} non trovato, skip {dataset_name}.")
            continue

        run_dataset(llm, dataset_name, dataset_path, args, llm_folder)  # args contiene max_new_tokens

    print("\nGenerazione completata!")


if __name__ == "__main__":
    main()
