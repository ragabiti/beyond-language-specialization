"""make_tables.py — Ristampa le tabelle del paper dai summary in results/.
Uso: python3 make_tables.py"""
import json
from pathlib import Path

JUDGES = {"gpt4o_judge": "GPT-4o-mini (primary)", "qwen_judge": "Qwen3-8B (secondary)"}
GROUPS = ["english", "uniqa", "pharmaqa"]

def fmt(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "  —  "

def ac_of(r):   # AC ha nome diverso tra i due collettori
    return r.get("accuracy", r.get("accuracy_ragas"))
def ar_of(r):
    return r.get("answer_relevance", r.get("ar_ragas_avg"))
def f_of(r):
    return r.get("faithfulness", r.get("faithfulness_avg"))
def cr_of(r):
    return r.get("context_relevance", r.get("context_relevance_avg"))

for judge, label in JUDGES.items():
    base = Path("results") / judge
    if not base.exists():
        continue
    print(f"\n{'='*72}\n  JUDGE: {label}\n{'='*72}")
    for g in GROUPS:
        f = base / f"{g}.json"
        if not f.exists():
            continue
        rows = json.loads(f.read_text(encoding="utf-8"))
        for ds in sorted({r["dataset"] for r in rows}):
            print(f"\n-- {g} / {ds} --")
            print(f"{'model':<42}{'AC':>8}{'F':>8}{'AR':>8}{'CR':>8}")
            for r in sorted((r for r in rows if r["dataset"] == ds),
                            key=lambda r: -(ac_of(r) or 0)):
                print(f"{r['model_id']:<42}{fmt(ac_of(r)):>8}{fmt(f_of(r)):>8}"
                      f"{fmt(ar_of(r)):>8}{fmt(cr_of(r)):>8}")