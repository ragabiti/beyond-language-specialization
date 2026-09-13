"""
collect_gpt.py — Unisce i summary GPT-4o-mini (AC da un file, F/AR/CR
dall'altro) in 3 file per-tabella, mappati su Tab. 1/2/3 della camera-ready.

Sola lettura sulla working dir; scrive solo in --out.

Per ogni gruppo passi la coppia:  <AC_file>:<RAGMETRICS_file>

Uso:
    python3 collect_gpt.py --out results/gpt4o_judge \
      --eng     WD/eng/answer_correctness_accuracy_by_model_dataset.json:WD/eng_rag_metrics/rag_metrics_summary_by_model_dataset.json \
      --uniqa   WD/uniqa/answer_correctness_accuracy_by_model_dataset.json:WD/uniqa_rag_metrics/rag_metrics_summary_by_model_dataset.json \
      --pharma  WD/pharmaqa/answer_correctness_accuracy_by_model_dataset.json:WD/pharmaqa_rag_metrics/rag_metrics_summary_by_model_dataset.json

(WD = ~/Desktop/WORK/VELVET/remake_PON/The-Power-of-Noise/data/llm_as_a_judge)
"""
import json, argparse
from pathlib import Path

EXCLUDED_MODELS = {"velvet-25b-011-p10"}  # non nel paper


def load(path):
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def merge_group(ac_path, rm_path):
    ac = load(ac_path)
    rm = load(rm_path)
    # indicizza F/AR/CR per (model, dataset)
    rmx = {(r["model_id"], r["dataset"]): r for r in rm}
    rows = []
    for a in ac:
        m, ds = a["model_id"], a["dataset"]
        if m in EXCLUDED_MODELS:
            continue
        r = rmx.get((m, ds), {})
        rows.append({
            "model_id": m,
            "dataset": ds,
            "accuracy": round(a.get("accuracy"), 4) if a.get("accuracy") is not None else None,
            "faithfulness": round(r["faithfulness"], 4) if r.get("faithfulness") is not None else None,
            "answer_relevance": round(r["answer_relevance"], 4) if r.get("answer_relevance") is not None else None,
            "context_relevance": round(r["context_relevance"], 4) if r.get("context_relevance") is not None else None,
            "n": a.get("n"),
        })
        if (m, ds) not in rmx:
            print(f"  ATTENZIONE: manca rag_metrics per {(m, ds)}")
    return sorted(rows, key=lambda r: (r["dataset"], -(r["accuracy"] or 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--eng")
    ap.add_argument("--uniqa")
    ap.add_argument("--pharma")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    groups = {"english": args.eng, "uniqa": args.uniqa, "pharmaqa": args.pharma}

    for name, pair in groups.items():
        if not pair:
            continue
        ac_path, rm_path = pair.split(":")
        rows = merge_group(ac_path, rm_path)
        outp = out_dir / f"{name}.json"
        outp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {name:<10} {len(rows):>2} righe -> {outp}  ({outp.stat().st_size/1024:.1f} kB)")
        for ds in sorted({r['dataset'] for r in rows}):
            print(f"  -- {ds} --")
            for r in [x for x in rows if x['dataset'] == ds]:
                print(f"     {r['model_id']:<42} AC={r['accuracy']}  F={r['faithfulness']}  "
                      f"AR={r['answer_relevance']}  CR={r['context_relevance']}")


if __name__ == "__main__":
    main()