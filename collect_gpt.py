"""
collect_gpt.py — Unisce i summary GPT-4o-mini (AC da un file, F/AR/CR
dall'altro) in 3 file per-tabella, mappati su Tab. 1/2/3 della camera-ready.

Sola lettura sulla working dir; scrive solo in --out.

Per ogni gruppo si passa la coppia:  <AC_file>:<RAGMETRICS_file>

I due file sorgente hanno strutture diverse:
  - AC_file contiene accuracy e n (answer correctness).
  - RAGMETRICS_file contiene faithfulness, answer_relevance e
    context_relevance.

I record vengono uniti per chiave (model_id, dataset) in un unico
dizionario con tutte e quattro le metriche.

Dipendenze:
  - json     : lettura / scrittura dei file JSON
  - argparse : parsing degli argomenti da riga di comando
  - pathlib  : gestione multipiattaforma dei percorsi

Uso:
    python3 collect_gpt.py --out results/gpt4o_judge \\
      --eng     WD/eng/answer_correctness_accuracy_by_model_dataset.json:WD/eng_rag_metrics/rag_metrics_summary_by_model_dataset.json \\
      --uniqa   WD/uniqa/answer_correctness_accuracy_by_model_dataset.json:WD/uniqa_rag_metrics/rag_metrics_summary_by_model_dataset.json \\
      --pharma  WD/pharmaqa/answer_correctness_accuracy_by_model_dataset.json:WD/pharmaqa_rag_metrics/rag_metrics_summary_by_model_dataset.json

(WD = ~/Desktop/WORK/VELVET/remake_PON/The-Power-of-Noise/data/llm_as_a_judge)
"""
import json, argparse
from pathlib import Path

# Modelli esclusi dal paper (non pubblicati nella versione finale)
EXCLUDED_MODELS = {"velvet-25b-011-p10"}


def load(path):
    """Carica e restituisce il contenuto di un file JSON.

    Args:
        path: Percorso (stringa o Path) del file JSON da leggere.
              Il carattere ~ viene espanso automaticamente.

    Returns:
        Oggetto Python (lista o dizionario) risultante dal parsing
        del file JSON.

    Raises:
        FileNotFoundError: Se il file non esiste.
        json.JSONDecodeError: Se il contenuto non e JSON valido.
    """
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def merge_group(ac_path, rm_path):
    """Unisce i risultati di answer correctness e metriche RAG per un gruppo.

    Legge i due file sorgente, indicizza le metriche RAG per chiave
    (model_id, dataset) e poi itera sui record AC per costruire righe
    unificate con tutte e quattro le metriche: accuracy, faithfulness,
    answer_relevance e context_relevance.

    Args:
        ac_path: Percorso del file JSON contenente le accuracy
                 (answer_correctness_accuracy_by_model_dataset.json).
        rm_path: Percorso del file JSON contenente le metriche RAG
                 (rag_metrics_summary_by_model_dataset.json).

    Returns:
        Lista di dizionari ordinata per (dataset, -accuracy). Ogni
        dizionario contiene: model_id, dataset, accuracy, faithfulness,
        answer_relevance, context_relevance, n.
    """
    # Caricamento dei due file sorgente
    ac = load(ac_path)
    rm = load(rm_path)

    # Indicizzazione delle metriche RAG per coppia (model_id, dataset)
    # per consentire un lookup O(1) durante la fusione
    rmx = {(r["model_id"], r["dataset"]): r for r in rm}

    rows = []
    # Iterazione sui record di answer correctness come base della fusione
    for a in ac:
        m, ds = a["model_id"], a["dataset"]

        # Esclusione dei modelli non presenti nel paper
        if m in EXCLUDED_MODELS:
            continue

        # Recupero del record RAG corrispondente (dizionario vuoto se assente)
        r = rmx.get((m, ds), {})

        # Costruzione della riga unificata con arrotondamento a 4 decimali
        rows.append({
            "model_id": m,
            "dataset": ds,
            "accuracy": round(a.get("accuracy"), 4) if a.get("accuracy") is not None else None,
            "faithfulness": round(r["faithfulness"], 4) if r.get("faithfulness") is not None else None,
            "answer_relevance": round(r["answer_relevance"], 4) if r.get("answer_relevance") is not None else None,
            "context_relevance": round(r["context_relevance"], 4) if r.get("context_relevance") is not None else None,
            "n": a.get("n"),
        })

        # Segnalazione a terminale se manca il record RAG per una coppia
        if (m, ds) not in rmx:
            print(f"  ATTENZIONE: manca rag_metrics per {(m, ds)}")

    # Ordinamento finale: per dataset e poi per accuracy decrescente
    return sorted(rows, key=lambda r: (r["dataset"], -(r["accuracy"] or 0)))


def main():
    """Punto di ingresso principale dello script.

    Fasi di esecuzione:
      1. Parsing degli argomenti (--out, --eng, --uniqa, --pharma).
      2. Per ciascun gruppo fornito, splitting del parametro nella
         coppia AC_file : RAGMETRICS_file.
      3. Fusione dei dati tramite merge_group.
      4. Scrittura del file JSON di output per ciascun gruppo.
      5. Stampa a terminale dei risultati per verifica visiva.

    Raises:
        SystemExit: Se l'argomento --out non e fornito.
    """
    # --- Parsing degli argomenti da riga di comando ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--eng")
    ap.add_argument("--uniqa")
    ap.add_argument("--pharma")
    args = ap.parse_args()

    # Creazione della cartella di output (se non esiste)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Mappatura nome-gruppo -> stringa con i due percorsi separati da ':'
    groups = {"english": args.eng, "uniqa": args.uniqa, "pharmaqa": args.pharma}

    # --- Ciclo sui gruppi: fusione e scrittura ---
    for name, pair in groups.items():
        # Se il gruppo non e stato specificato, lo salta
        if not pair:
            continue

        # Splitting della stringa in percorso AC e percorso metriche RAG
        ac_path, rm_path = pair.split(":")

        # Fusione dei dati dei due file sorgente
        rows = merge_group(ac_path, rm_path)

        # Scrittura del file JSON di output per il gruppo corrente
        outp = out_dir / f"{name}.json"
        outp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {name:<10} {len(rows):>2} righe -> {outp}  ({outp.stat().st_size/1024:.1f} kB)")

        # --- Stampa a terminale per verifica visiva ---
        for ds in sorted({r['dataset'] for r in rows}):
            print(f"  -- {ds} --")
            for r in [x for x in rows if x['dataset'] == ds]:
                print(f"     {r['model_id']:<42} AC={r['accuracy']}  F={r['faithfulness']}  "
                      f"AR={r['answer_relevance']}  CR={r['context_relevance']}")


if __name__ == "__main__":
    main()
