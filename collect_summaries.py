"""
collect_summaries.py — Estrae i summary RAGAS (Qwen3-8B judge) dai
ragass_results.json della working dir e produce 3 file piccoli, uno
per gruppo di dataset, mappati 1:1 sulle Tabelle 5/6/7 del paper.

NON tocca la working dir: solo lettura. Scrive esclusivamente in --out.

Mappa run -> tabella (verificata contro la camera-ready):
  triviaqa, nq                 -> results_qwen3       (eccetto Llama, Villanova)
  bioasq                       -> results_bioasq_26_5 (eccetto Llama, Villanova)
  triviaqa, nq, bioasq (Llama) -> results_llama_26_5
  *, Villanova                 -> results_villanova_eng
  pharmaqa_it                  -> results_pharmaqa_21_5
  uniqa_it                     -> results_uniqa_25_5

Uso:
    python3 tools/collect_summaries.py \
        --source ~/Desktop/WORK/VELVET/remake_PON/The-Power-of-Noise/data/ragass \
        --out results/qwen_judge
"""

import json
import glob
import argparse
from pathlib import Path


ENG_DATASETS = {"triviaqa", "nq", "bioasq"}
EXCLUDED_MODELS = {"velvet-25b-011-p10"}   

def winning_run(model_id: str, dataset: str) -> str:
    m = model_id.lower()
    if "villanova" in m and dataset in ENG_DATASETS:
        return "results_villanova_eng"
    if "llama" in m and dataset in ENG_DATASETS:
        return "results_llama_26_5"
    if dataset == "bioasq":
        return "results_bioasq_26_5"
    if dataset in ("triviaqa", "nq"):
        return "results_qwen3"
    if dataset == "pharmaqa_it":
        return "results_pharmaqa_21_5"
    if dataset == "uniqa_it":
        return "results_uniqa_25_5"
    return ""


def dataset_group(dataset: str) -> str:
    if dataset in ENG_DATASETS:
        return "english"
    if dataset == "pharmaqa_it":
        return "pharmaqa"
    if dataset == "uniqa_it":
        return "uniqa"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                     help="Cartella data/ragass della working dir (sola lettura)")
    ap.add_argument("--out", required=True,
                     help="Cartella di destinazione nel repo (verrà creata)")
    args = ap.parse_args()

    source = Path(args.source).expanduser()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(source / "results_*" / "*" / "*" / "ragass_results.json")
    files = glob.glob(pattern)
    print(f"Trovati {len(files)} ragass_results.json in {source}")

    kept = {"english": [], "uniqa": [], "pharmaqa": []}
    seen = set()

    for f in sorted(files):
        run = Path(f).relative_to(source).parts[0]
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        s = data.get("summary", {})
        model_id, dataset = s.get("model_id"), s.get("dataset")
        if not model_id or not dataset:
            continue
        if model_id in EXCLUDED_MODELS:        # ← nuova riga
            continue                            # ← nuova riga

        want = winning_run(model_id, dataset)

        want = winning_run(model_id, dataset)
        if run != want:
            continue  # non è la run designata per questa cella -> scarta

        key = (model_id, dataset)
        if key in seen:
            print(f"  ATTENZIONE: doppia riga vincente per {key} (run={run})")
            continue
        seen.add(key)

        group = dataset_group(dataset)
        kept[group].append(s)  # solo il summary, MAI i details

    for group, rows in kept.items():
        out_path = out_dir / f"{group}.json"
        rows_sorted = sorted(rows, key=lambda r: (r["dataset"], r["model_id"]))
        out_path.write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        print(f"  {group:<10} {len(rows_sorted):>3} righe -> {out_path}  "
              f"({out_path.stat().st_size/1024:.1f} kB)")

    # Stampa le tabelle per verifica visiva contro la camera-ready
    for group, rows in kept.items():
        if not rows:
            continue
        print(f"\n=== {group.upper()} ===")
        for ds in sorted({r["dataset"] for r in rows}):
            print(f"-- {ds} --")
            for r in sorted((r for r in rows if r["dataset"] == ds),
                             key=lambda r: -(r.get("accuracy_ragas") or 0)):
                print(f"  {r['model_id']:<42} AC={r.get('accuracy_ragas')}  "
                      f"F={r.get('faithfulness_avg')}  AR={r.get('ar_ragas_avg')}  "
                      f"CR={r.get('context_relevance_avg')}")


if __name__ == "__main__":
    main()