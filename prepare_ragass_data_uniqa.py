"""
prepare_ragass_data_uniqa.py

Prepara i dati per la valutazione RAGAS su UniQA IT.

Questo script costruisce il file JSON usato da run_ragass_phase1_uniqa.py.
Per ogni modello carica i risultati generati nei file .pkl e aggiunge il
contesto RAG necessario per calcolare faithfulness e context relevance.

Scelte metodologiche:
  - Dataset: uniqa_it.
  - Sample: tutti gli example_id comuni ai modelli disponibili, salvo N_MAX.
  - Contesto: ricostruito da UniQA/data/uniqa_it_dataset.json, non dai .pkl.
    I .pkl UniQA salvano domanda, gold answers, risposta generata e metadati,
    ma non salvano direttamente i testi dei passage recuperati.
  - Il contesto usato per RAGAS è il top-5 ordinato per rank del dataset
    UniQA già arricchito con retrieval FAISS/BGE.
  - Questa scelta è corretta solo se UniQA/data/uniqa_it_dataset.json è lo
    stesso file usato durante la generazione delle risposte.
  - Tutti i modelli elencati sono valutati su top5.

Output:
    data/ragass/ragass_samples_uniqa_it.json

Lanciare dalla root del progetto:
    cd ~/Desktop/WORK/VELVET/remake_PON/The-Power-of-Noise
    conda activate ragass
    python prepare_ragass_data_uniqa.py
"""

import pickle
import json
import random
import argparse
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
SEED    = 42
N_MAX   = None   # None = usa tutti i sample disponibili; int = limita a N_MAX

GEN_RES_DIR = Path("data/gen_res")
OUT_PATH    = Path("data/ragass/ragass_samples_uniqa_it.json")

# Modelli e relativo top_k usato per la valutazione.
# Tutti i modelli qui sotto vengono valutati su top5.
MODELS = {
    "google/gemma-4-E2B-it":                5,
    "ibm-granite/granite-4.1-3B":           5,
    "Qwen/Qwen3.5-9B":                      5,
    "ibm-granite/granite-4.1-8b":           5,
    "mistralai/Ministral-8B-Instruct-2410": 5,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 5,
    "Fastweb/FastwebMIIA-7B":               5,
    "VillanovaAI/Villanova-2B-2603":        5,
    "velvet-25b-011-p10":                   5,
    "velvet-14b":                           5,
    "velvet-2b-1.5-03-23918":              5,
}

DATASET = "uniqa_it"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def model_slug(model_id: str) -> str:
    return model_id.split("/")[-1] if "/" in model_id else model_id


def load_pkl_for_model(model_id: str, top_k: int) -> dict:
    """
    Carica tutti i pkl di un modello su uniqa_it.
    Ritorna {example_id: record}.
    """
    base = GEN_RES_DIR / model_slug(model_id) / DATASET / f"top{top_k}"
    if not base.exists():
        raise FileNotFoundError(f"Directory non trovata: {base}")

    records = {}
    for pkl_file in sorted(base.glob(f"results_top{top_k}_info_*.pkl")):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        for item in data:
            records[item["example_id"]] = item

    if not records:
        raise ValueError(f"Nessun record trovato in {base}")
    return records


def build_context_from_record(rec: dict, top_k: int) -> str:
    """
    Fallback non usato nel flusso principale.

    I .pkl UniQA attuali non contengono i testi dei passage recuperati.
    Per questo motivo il contesto viene ricostruito da load_uniqa_passages(),
    usando UniQA/data/uniqa_it_dataset.json.
    """
    return rec.get("context", "")


def load_uniqa_passages() -> dict:
    """
    Carica i passage recuperati per UniQA IT.

    Il file UniQA/data/uniqa_it_dataset.json deve essere quello prodotto
    dopo il retrieval FAISS/BGE: ogni esempio contiene 25 passage ordinabili
    tramite il campo "rank". Per RAGAS viene usato il top5.

    Ritorna:
        {example_id: context_string}
    """
    dataset_path = Path("UniQA/data/uniqa_it_dataset.json")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        raw = json.load(f)

    passages_map = {}
    for ex in raw:
        eid = str(ex["example_id"])
        # Prendi i passage recuperati, ordinati per rank, e usa il top5.
        sorted_p = sorted(ex["passages"], key=lambda p: p["rank"])
        context_str = "\n\n".join(
            f"[Document {i+1}] {p['text']}"
            for i, p in enumerate(sorted_p[:5])
        )
        passages_map[eid] = context_str

    return passages_map


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=str, default=None,
                        help=f"Path del file JSON di output (default: {OUT_PATH})")
    args = parser.parse_args()

    out_path = Path(args.output_file) if args.output_file else OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    print(f"\n{'='*55}")
    print(f"  Dataset: {DATASET.upper()}")
    print(f"{'='*55}")

    # 1. Carica pkl per ogni modello
    model_records = {}
    for m, top_k in MODELS.items():
        try:
            recs = load_pkl_for_model(m, top_k)
            model_records[m] = recs
            print(f"  ✓ {model_slug(m):<40} {len(recs)} esempi  (top{top_k})")
        except Exception as e:
            print(f"  ✗ {model_slug(m):<40} SKIP ({e})")

    if not model_records:
        print("  Nessun modello disponibile.")
        return

    # 2. Intersezione example_id comuni a tutti i modelli presenti
    common_ids = set.intersection(
        *[set(r.keys()) for r in model_records.values()]
    )
    print(f"\n  example_id comuni a tutti i modelli: {len(common_ids)}")

    # 3. Campionamento (tutti o N_MAX)
    if N_MAX is not None:
        n_sample = min(N_MAX, len(common_ids))
        sampled_ids = rng.sample(sorted(common_ids), n_sample)
        print(f"  Campionati (N_MAX={N_MAX}): {len(sampled_ids)}")
    else:
        sampled_ids = sorted(common_ids)
        print(f"  Uso tutti: {len(sampled_ids)}")

    # 4. Carica i passage recuperati dal dataset UniQA arricchito con retrieval.
    print(f"\n  Caricamento passaggi da UniQA/data/uniqa_it_dataset.json...")
    try:
        passages_map = load_uniqa_passages()
        print(f"  Passages caricati: {len(passages_map)} esempi")
    except FileNotFoundError as e:
        print(f"  ⚠️  {e}")
        print(f"  Il contesto sarà vuoto — Faithfulness e CR non calcolabili")
        passages_map = {}

    # 5. Costruisce i record RAGAS per ogni coppia modello × sample.
    output = []
    for eid in sampled_ids:
        context_str = passages_map.get(str(eid), "")

        for m, records in model_records.items():
            rec = records[eid]
            output.append({
                "example_id":        eid,
                "dataset":           DATASET,
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
    print(f"  Ora lancia: python run_ragass_phase1_uniqa.py")


if __name__ == "__main__":
    main()
