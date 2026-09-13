"""
Generazione risposte RAG su PharmaQA.IT con modelli Velvet.
Adattato da generate_rag_velvet.py.

Uso:
    python generate_rag_velvet_pharmaqa.py --llm_id velvet-25b-011-p10
    python generate_rag_velvet_pharmaqa.py --llm_id velvet-14b
    python generate_rag_velvet_pharmaqa.py --llm_id velvet-2b-1.5-03-23918

Output:
    data/gen_res/<model>/pharmaqa_it/top5/
"""
import re
import os
import sys
import json
import pickle
import argparse
from tqdm import tqdm
sys.path.insert(0, "src")
from llm import LLM


DATASET_PATHS = {
    'pharmaqa_it': 'UniQA/data/dataset/pharmaqa_dataset.json',
}

TASK_INSTRUCTION = (
    "Sei un assistente farmaceutico esperto. Ti vengono forniti estratti di un "
    "Riassunto delle Caratteristiche del Prodotto (RCP) e una domanda. "
    "Rispondi ESTRAENDO la risposta (massimo 10 token) dal documento, "
    "basandoti ESCLUSIVAMENTE sulle informazioni presenti. "
    "Se il documento non contiene la risposta, rispondi con NO-RES."
)


def build_prompt(question: str, passages: list) -> str:
    docs_str = "\n".join(
        f"Document [{i+1}](Title: passage) {p['text']}"
        for i, p in enumerate(passages)
    )
    return f"{TASK_INSTRUCTION}\nDocuments:\n{docs_str}\nQuestion: {question}\nAnswer:"


def string_match(generated: str, answers: list) -> bool:
    gen = generated.strip().lower()
    for sep in ['\n', 'Document [', '(', 'Question:']:
        if sep in gen:
            gen = gen.split(sep)[0]
    gen = gen.strip()
    return any(ans.lower().strip() in gen for ans in answers)

def clean_generated(output: str) -> str:
    if "Answer:" in output:
        output = output[output.find("Answer:") + len("Answer:"):]
    # Fix Qwen: estrai dopo </think>
    if '</think>' in output:
        output = output[output.rfind('</think>') + len('</think>'):]
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
    for stop in ["Question:", "Document [", "Human:", "User:", "\nAnswer:"]:
        if stop in output:
            output = output[:output.find(stop)]
    return output.strip()

def run_dataset(llm, dataset_name: str, dataset_path: str, args, llm_folder: str):
    print(f"\n{'#'*60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'#'*60}")

    with open(dataset_path, encoding='utf-8') as f:
        data = json.load(f)

    if args.max_samples:
        data = data[:args.max_samples]

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
            prompt   = build_prompt(example['question'], passages)

            output = llm.generate(prompt, max_new_tokens=args.max_new_tokens)
            if isinstance(output, list):
                output = output[0]

            generated = clean_generated(output)

            is_correct = string_match(generated, example['answers'])
            if is_correct:
                correct += 1

            results.append({
                'example_id':  example['example_id'],
                'question':    example['question'],
                'answers':     example['answers'],
                'generated':   generated,
                'is_correct':  is_correct,
                'k':           k,
                'dataset':     dataset_name,
                'prompt_temp': example.get('prompt_temp', ''),
                'n_relevant':  sum(1 for p in passages if p.get('is_relevant', False)),
            })

            if (idx + 1) % args.save_every == 0 or (idx + 1) == len(data):
                acc = correct / (idx + 1)
                print(f"  [{idx+1}/{len(data)}] Accuracy running: {acc:.4f}")
                fname = os.path.join(save_dir, f"results_top{k}_info_{idx+1}.pkl")
                with open(fname, 'wb') as f:
                    pickle.dump(results, f)

        final_acc = correct / len(data)
        print(f"\n{dataset_name} k={k} — Accuracy: {final_acc:.4f}")

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
    parser.add_argument('--datasets',       type=str, nargs='+', default=['pharmaqa_it'])
    parser.add_argument('--k_values',       type=int, nargs='+', default=[5])
    parser.add_argument('--output_dir',     type=str, default='data/gen_res')
    parser.add_argument('--save_every',     type=int, default=100)
    parser.add_argument('--max_samples',    type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--overwrite',      action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Modello:        {args.llm_id}")
    print(f"Dataset:        {args.datasets}")
    print(f"k values:       {args.k_values}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"Output:         {args.output_dir}")

    llm = LLM(args.llm_id)
    llm_folder = args.llm_id.split("/")[-1] if '/' in args.llm_id else args.llm_id

    for dataset_name in args.datasets:
        dataset_path = DATASET_PATHS[dataset_name]
        if not os.path.exists(dataset_path):
            print(f"\nATTENZIONE: {dataset_path} non trovato, skip.")
            continue
        run_dataset(llm, dataset_name, dataset_path, args, llm_folder)

    print("\nGenerazione completata!")


if __name__ == "__main__":
    main()