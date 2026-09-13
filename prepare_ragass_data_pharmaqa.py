"""
prepare_ragass_data_pharmaqa.py

Prepara i dati per la valutazione RAGAS su PharmaQA.IT.
Versione adattata di prepare_ragass_data_eng.py per dataset italiano.

Cosa fa:
  1. Carica i pkl di tutti i modelli per pharmaqa_it
  2. Trova gli example_id comuni a tutti i modelli
  3. Campiona N_SAMPLES esempi con seed fisso (riproducibile)
  4. Ricostruisce il contesto top-k da UniQA/data/pharmaqa/retrieved_top5.json
  5. Salva tutto in un unico file JSON flat per run_ragass_phase1.py

Lanciare dalla root del progetto:
    cd ~/Desktop/WORK/VELVET/remake_PON/The-Power-of-Noise
    conda activate ragass
    python prepare_ragass_data_pharmaqa.py --output-file data/ragass/ragass_samples_pharmaqa.json
"""

import pickle
import json
import random
import argparse
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
SEED      = 42
TOP_K     = 5
N_SAMPLES = 1000  # campiona fino a N esempi (ridotto se common_ids < N)

# k per modello — default TOP_K, override per modelli con context window limitata
MODEL_K = {
    "sapienzanlp/Minerva-7B-instruct-v1.0": 3,  # LLaMA-2 base: 4096 token context
}

GEN_RES_DIR   = Path("data/gen_res")
PHARMAQA_PATH = Path("data/dataset/pharmaqa_dataset.json")
OUT_PATH      = Path("data/ragass/ragass_samples_pharmaqa.json")

DATASETS = ["pharmaqa_it"]

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


def load_pkl_for_model_dataset(model_id: str, dataset: str, k: int) -> dict:
    """
    Carica tutti i pkl di un modello/dataset.
    Ritorna {example_id: record}.
    """
    base = GEN_RES_DIR / model_slug(model_id) / dataset / f"top{k}"
    if not base.exists():
        raise FileNotFoundError(f"Directory non trovata: {base}")

    records = {}
    for pkl_file in sorted(base.glob(f"results_top{k}_info_*.pkl")):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        for item in data:
            records[item["example_id"]] = item

    if not records:
        raise ValueError(f"Nessun record trovato in {base}")
    return records


def load_passages() -> dict:
    """
    Carica i passage da UniQA/data/pharmaqa/retrieved_top5.json.
    Ritorna {example_id: [testo_p0, ..., testo_p(k-1)]}.
    """
    if not PHARMAQA_PATH.exists():
        raise FileNotFoundError(f"Dataset non trovato: {PHARMAQA_PATH}")

    with open(PHARMAQA_PATH, encoding="utf-8") as f:
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

    dataset = "pharmaqa_it"
    print(f"\n{'='*55}")
    print(f"  Dataset: PHARMAQA_IT  (target: {N_SAMPLES} sample)")
    print(f"{'='*55}")

    # 1. Carica pkl per ogni modello
    model_records = {}
    model_k = {}
    for m in MODELS:
        k = MODEL_K.get(m, TOP_K)
        try:
            recs = load_pkl_for_model_dataset(m, dataset, k)
            model_records[m] = recs
            model_k[m] = k
            note = f" [k={k}]" if k != TOP_K else ""
            print(f"  ✓ {m:<45} {len(recs)} esempi{note}")
        except Exception as e:
            print(f"  ✗ {m:<45} SKIP ({e})")

    if not model_records:
        print("  Nessun modello disponibile.")
        return

    # 2. Intersezione example_id comuni a tutti i modelli presenti
    common_ids = set.intersection(
        *[set(r.keys()) for r in model_records.values()]
    )
    print(f"\n  example_id comuni: {len(common_ids)}")

    n_sample = min(N_SAMPLES, len(common_ids))
    if n_sample < N_SAMPLES:
        print(f"  ⚠️  Meno di {N_SAMPLES} esempi comuni, uso {n_sample}")

    # 3. Campionamento riproducibile
    sampled_ids = rng.sample(sorted(common_ids), n_sample)
    print(f"  Campionati: {len(sampled_ids)}")

    # 4. Carica passages
    try:
        passages_map = load_passages()
        print(f"  Passages caricati: {len(passages_map)} esempi")
    except FileNotFoundError as e:
        print(f"  ⚠️  {e}")
        print(f"  Il contesto sarà vuoto (Faithfulness non calcolabile)")
        passages_map = {}

    # 5. Costruisce un record per ogni modello × sample
    for eid in sampled_ids:
        for m, records in model_records.items():
            k = model_k[m]
            context_passages = passages_map.get(str(eid), [])[:k]
            context_str = "\n\n".join(
                f"[Document {i+1}] {p}"
                for i, p in enumerate(context_passages)
            )
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
    n_models  = len(set(r["model_id"] for r in output))
    n_samples = len(set(r["example_id"] for r in output))

    print(f"\n{'='*55}")
    print(f"  Totale record:  {len(output)}")
    print(f"  Modelli:        {n_models}")
    print(f"  Sample unici:   {n_samples}")
    print(f"  Attesi totali:  {n_samples * n_models}")
    print(f"{'='*55}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Salvato in: {out_path}")


if __name__ == "__main__":
    main()
