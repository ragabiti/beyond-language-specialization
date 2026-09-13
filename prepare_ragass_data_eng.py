"""
prepare_ragass_data.py

Prepara i dati per la valutazione RAGAS partendo dai pkl già generati.

Cosa fa:
  1. Carica i pkl di tutti i modelli per ogni dataset
  2. Trova gli example_id comuni a tutti i modelli
  3. Campiona N_PER_DS esempi per dataset con seed fisso (riproducibile)
  4. Ricostruisce il contesto top-k dai dataset originali
  5. Salva tutto in un unico file JSON flat per run_ragass.py

Output: data/ragass/ragass_samples.json
        → 11 modelli × 1000 sample = 11.000 record

Lanciare dalla root del progetto:
    cd ~/Desktop/WORK/VELVET/remake_PON/The-Power-of-Noise
    conda activate ragass
    python prepare_ragass_data.py
"""

import pickle
import json
import random
import argparse
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
SEED  = 42
TOP_K = 5

# 334 + 333 + 333 = 1000 sample totali
N_PER_DS = {
    "triviaqa": 334,
    "nq":       333,
    "bioasq":   333,
}

# Path relativi alla root del progetto
GEN_RES_DIR = Path("data/gen_res")
DATASET_DIR = Path("data/dataset")
OUT_PATH    = Path("data/ragass/ragass_samples_eng.json")

DATASETS = ["triviaqa", "nq", "bioasq"]

MODELS = [
    "google/gemma-4-E2B-it",
    "ibm-granite/granite-4.1-3B",
    "Qwen/Qwen3.5-9B",
    "ibm-granite/granite-4.1-8b",
    "mistralai/Ministral-8B-Instruct-2410",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Fastweb/FastwebMIIA-7B",
    "sapienzanlp/Minerva-7B-instruct-v1.0",
    "VillanovaAI/Villanova-2B-2603",
    "velvet-25b-011-p10",
    "velvet-14b",
    "velvet-2b-1.5-03-23918",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def model_slug(model_id: str) -> str:
    """
    Converte il model_id nel nome della cartella in data/gen_res/.
    Le cartelle usano solo il nome del modello senza il prefisso organizzazione.
      google/gemma-4-E2B-it       → gemma-4-E2B-it
      ibm-granite/granite-4.1-3B  → granite-4.1-3B
      Qwen/Qwen3.5-9B             → Qwen3.5-9B
      velvet-25b-011-p10          → velvet-25b-011-p10
    """
    return model_id.split("/")[-1] if "/" in model_id else model_id


def load_pkl_for_model_dataset(model_id: str, dataset: str) -> dict:
    """
    Carica tutti i pkl di un modello/dataset.
    Ritorna {example_id: record}.
    """
    base = GEN_RES_DIR / model_slug(model_id) / dataset / f"top{TOP_K}"
    if not base.exists():
        raise FileNotFoundError(f"Directory non trovata: {base}")

    records = {}
    for pkl_file in sorted(base.glob(f"results_top{TOP_K}_info_*.pkl")):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        for item in data:
            records[item["example_id"]] = item

    if not records:
        raise ValueError(f"Nessun record trovato in {base}")
    return records


def load_passages(dataset: str) -> dict:
    """
    Carica i passage originali dal dataset JSON.
    Ritorna {example_id: [testo_p0, ..., testo_p(k-1)]}.
    Cerca prima <dataset>_dataset.json, poi <dataset>.json.
    """
    candidates = [
        DATASET_DIR / f"{dataset}_dataset.json",
        DATASET_DIR / f"{dataset}.json",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Dataset non trovato. Cercato: {[str(c) for c in candidates]}"
        )

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    passages = {}
    for ex in raw:
        eid = str(ex["example_id"])
        sorted_p = sorted(ex["passages"], key=lambda p: p["rank"])
        passages[eid] = [p["text"] for p in sorted_p[:TOP_K]]
    return passages


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=str, default=None,
                        help=f"Path del file JSON di output (default: {OUT_PATH})")
    args = parser.parse_args()

    out_path = Path(args.output_file) if args.output_file else OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    output = []

    for dataset in DATASETS:
        n_target = N_PER_DS[dataset]
        print(f"\n{'='*55}")
        print(f"  Dataset: {dataset.upper()}  (target: {n_target} sample)")
        print(f"{'='*55}")

        # 1. Carica pkl per ogni modello
        model_records = {}
        for m in MODELS:
            try:
                recs = load_pkl_for_model_dataset(m, dataset)
                model_records[m] = recs
                print(f"  ✓ {m:<45} {len(recs)} esempi")
            except Exception as e:
                print(f"  ✗ {m:<45} SKIP ({e})")

        if not model_records:
            print("  Nessun modello disponibile, skip dataset.")
            continue

        # 2. Intersezione example_id comuni a tutti i modelli presenti
        common_ids = set.intersection(
            *[set(r.keys()) for r in model_records.values()]
        )
        print(f"\n  example_id comuni: {len(common_ids)}")

        n_sample = min(n_target, len(common_ids))
        if n_sample < n_target:
            print(f"  ⚠️  Meno di {n_target} esempi comuni, uso {n_sample}")

        # 3. Campionamento riproducibile (sorted per determinismo)
        sampled_ids = rng.sample(sorted(common_ids), n_sample)
        print(f"  Campionati: {len(sampled_ids)}")

        # 4. Carica passage originali
        try:
            passages_map = load_passages(dataset)
            print(f"  Passages caricati: {len(passages_map)} esempi")
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}")
            print(f"  Il contesto sarà vuoto (Faithfulness non calcolabile)")
            passages_map = {}

        # 5. Costruisce un record per ogni modello × sample
        for eid in sampled_ids:
            context_passages = passages_map.get(str(eid), [])
            context_str = "\n\n".join(
                f"[Document {i+1}] {p}"
                for i, p in enumerate(context_passages)
            )

            for m, records in model_records.items():
                rec = records[eid]
                output.append({
                    "example_id":        eid,
                    "dataset":           dataset,
                    "model_id":          m,
                    "question":          rec["question"],
                    "answers":           rec["answers"],
                    "generated":         rec["generated"],
                    "context":           context_str,
                    "is_correct_string": rec["is_correct"],
                    "n_relevant":        rec.get("n_relevant"),
                })

    # ── Riepilogo ─────────────────────────────────────────────────────────
    n_models   = len(set(r["model_id"] for r in output))
    n_datasets = len(set(r["dataset"]  for r in output))
    n_samples  = len(set((r["example_id"], r["dataset"]) for r in output))

    print(f"\n{'='*55}")
    print(f"  Totale record:  {len(output)}")
    print(f"  Modelli:        {n_models}")
    print(f"  Dataset:        {n_datasets}")
    print(f"  Sample unici:   {n_samples} (attesi: {sum(N_PER_DS.values())})")
    print(f"  Attesi totali:  {sum(N_PER_DS.values()) * n_models}")
    print(f"{'='*55}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Salvato in: {out_path}")


if __name__ == "__main__":
    main()
