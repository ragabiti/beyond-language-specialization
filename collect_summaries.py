"""
collect_summaries.py — Estrae i summary RAGAS (Qwen3-8B judge) dai
ragass_results.json della working dir e produce 3 file piccoli, uno
per gruppo di dataset, mappati 1:1 sulle Tabelle 5/6/7 del paper.

NON tocca la working dir: solo lettura. Scrive esclusivamente in --out.

Dataset supportati:
  - Inglese: triviaqa, nq, bioasq
  - Italiano: pharmaqa_it, uniqa_it

Modelli: qualsiasi modello presente nei file ragass_results.json, ad
eccezione di quelli elencati in EXCLUDED_MODELS.

Mappa run -> tabella (verificata contro la camera-ready):
  triviaqa, nq                 -> results_qwen3       (eccetto Llama, Villanova)
  bioasq                       -> results_bioasq_26_5 (eccetto Llama, Villanova)
  triviaqa, nq, bioasq (Llama) -> results_llama_26_5
  *, Villanova                 -> results_villanova_eng
  pharmaqa_it                  -> results_pharmaqa_21_5
  uniqa_it                     -> results_uniqa_25_5

Dipendenze:
  - json     : lettura / scrittura dei file JSON
  - glob     : ricerca ricorsiva dei file ragass_results.json
  - pathlib  : gestione multipiattaforma dei percorsi
  - argparse : parsing degli argomenti da riga di comando

Uso:
    python3 tools/collect_summaries.py \\
        --source ~/Desktop/WORK/VELVET/remake_PON/The-Power-of-Noise/data/ragass \\
        --out results/qwen_judge
"""

import json
import glob
import argparse
from pathlib import Path


ENG_DATASETS = {"triviaqa", "nq", "bioasq"}
EXCLUDED_MODELS = {"velvet-25b-011-p10"}

def winning_run(model_id: str, dataset: str) -> str:
    """Determina la run vincente per una data coppia (modello, dataset).

    La logica di mappatura segue l'ordine di priorita:
      1. Se il modello contiene 'villanova' e il dataset e inglese,
         restituisce la run Villanova dedicata.
      2. Se il modello contiene 'llama' e il dataset e inglese,
         restituisce la run Llama dedicata.
      3. Se il dataset e 'bioasq', restituisce la run bioasq.
      4. Se il dataset e 'triviaqa' o 'nq', restituisce la run qwen3.
      5. Se il dataset e 'pharmaqa_it' o 'uniqa_it', restituisce la
         run corrispondente al dataset italiano.
      6. Per qualsiasi altro caso restituisce stringa vuota (nessuna run).

    Args:
        model_id: Identificativo del modello (es. 'llama-3.1-8b').
        dataset:  Nome del dataset (es. 'triviaqa', 'bioasq', 'uniqa_it').

    Returns:
        Nome della cartella-run che contiene i risultati validi per la
        cella corrispondente nella tabella del paper, oppure stringa
        vuota se nessuna run corrisponde.
    """
    # Confronto case-insensitive sul nome del modello
    m = model_id.lower()

    # Priorita 1: modelli Villanova su dataset inglesi -> run dedicata
    if "villanova" in m and dataset in ENG_DATASETS:
        return "results_villanova_eng"

    # Priorita 2: modelli Llama su dataset inglesi -> run dedicata
    if "llama" in m and dataset in ENG_DATASETS:
        return "results_llama_26_5"

    # Priorita 3: dataset bioasq (tutti i modelli rimanenti)
    if dataset == "bioasq":
        return "results_bioasq_26_5"

    # Priorita 4: dataset triviaqa / nq -> run qwen3 generica
    if dataset in ("triviaqa", "nq"):
        return "results_qwen3"

    # Priorita 5: dataset italiani -> run specifiche per ciascuno
    if dataset == "pharmaqa_it":
        return "results_pharmaqa_21_5"
    if dataset == "uniqa_it":
        return "results_uniqa_25_5"

    # Nessuna run valida trovata
    return ""


def dataset_group(dataset: str) -> str:
    """Classifica un dataset nel gruppo di appartenenza per la tabella.

    La classificazione determina in quale file di output verra salvato
    il summary: english.json, pharmaqa.json oppure uniqa.json.

    Args:
        dataset: Nome del dataset (es. 'triviaqa', 'pharmaqa_it').

    Returns:
        Stringa che identifica il gruppo: 'english', 'pharmaqa',
        'uniqa' oppure 'other' se il dataset non rientra in nessuno
        dei tre gruppi attesi.
    """
    # Dataset inglesi: triviaqa, nq, bioasq
    if dataset in ENG_DATASETS:
        return "english"

    # Dataset italiani: ognuno costituisce un gruppo a se
    if dataset == "pharmaqa_it":
        return "pharmaqa"
    if dataset == "uniqa_it":
        return "uniqa"

    # Gruppo residuo per eventuali dataset non previsti
    return "other"


def main():
    """Punto di ingresso principale dello script.

    Fasi di esecuzione:
      1. Parsing degli argomenti --source e --out.
      2. Ricerca ricorsiva di tutti i ragass_results.json nella
         cartella sorgente tramite glob.
      3. Per ogni file trovato, verifica che la run corrisponda a
         quella vincente per la coppia (modello, dataset).
      4. Deduplicazione: ogni coppia (modello, dataset) puo comparire
         al massimo una volta.
      5. Scrittura dei file JSON di output, uno per gruppo.
      6. Stampa delle tabelle a terminale per verifica visiva.

    Raises:
        SystemExit: Se gli argomenti obbligatori non sono forniti.
    """
    # --- Parsing degli argomenti da riga di comando ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                     help="Cartella data/ragass della working dir (sola lettura)")
    ap.add_argument("--out", required=True,
                     help="Cartella di destinazione nel repo (verrà creata)")
    args = ap.parse_args()

    # Risoluzione dei percorsi sorgente e destinazione
    source = Path(args.source).expanduser()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Ricerca dei file ragass_results.json con pattern a 3 livelli ---
    # Il pattern corrisponde a: results_<run>/<modello>/<dataset>/ragass_results.json
    pattern = str(source / "results_*" / "*" / "*" / "ragass_results.json")
    files = glob.glob(pattern)
    print(f"Trovati {len(files)} ragass_results.json in {source}")

    # Dizionario dei risultati filtrati, indicizzato per gruppo
    kept = {"english": [], "uniqa": [], "pharmaqa": []}
    # Set per il controllo di deduplicazione
    seen = set()

    # --- Ciclo principale: filtraggio e selezione dei summary ---
    for f in sorted(files):
        # Estrazione del nome della run dal primo livello del percorso relativo
        run = Path(f).relative_to(source).parts[0]

        # Lettura e parsing del file JSON
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        s = data.get("summary", {})
        model_id, dataset = s.get("model_id"), s.get("dataset")

        # Scarta file privi di model_id o dataset nel summary
        if not model_id or not dataset:
            continue

        # Scarta i modelli esplicitamente esclusi dal paper
        if model_id in EXCLUDED_MODELS:
            continue

        # Verifica che la run corrente sia quella vincente per questa coppia
        want = winning_run(model_id, dataset)

        want = winning_run(model_id, dataset)
        if run != want:
            continue  # non è la run designata per questa cella -> scarta

        # --- Deduplicazione: ogni coppia (modello, dataset) compare una sola volta ---
        key = (model_id, dataset)
        if key in seen:
            print(f"  ATTENZIONE: doppia riga vincente per {key} (run={run})")
            continue
        seen.add(key)

        # Aggiunta del summary al gruppo corrispondente (solo summary, MAI details)
        group = dataset_group(dataset)
        kept[group].append(s)  # solo il summary, MAI i details

    # --- Scrittura dei file JSON di output, uno per gruppo ---
    for group, rows in kept.items():
        out_path = out_dir / f"{group}.json"
        # Ordinamento per dataset e poi per model_id per coerenza
        rows_sorted = sorted(rows, key=lambda r: (r["dataset"], r["model_id"]))
        out_path.write_text(json.dumps(rows_sorted, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        print(f"  {group:<10} {len(rows_sorted):>3} righe -> {out_path}  "
              f"({out_path.stat().st_size/1024:.1f} kB)")

    # --- Stampa le tabelle per verifica visiva contro la camera-ready ---
    for group, rows in kept.items():
        if not rows:
            continue
        print(f"\n=== {group.upper()} ===")
        # Iterazione per dataset all'interno di ciascun gruppo
        for ds in sorted({r["dataset"] for r in rows}):
            print(f"-- {ds} --")
            # Ordinamento decrescente per accuracy_ragas per evidenziare il modello migliore
            for r in sorted((r for r in rows if r["dataset"] == ds),
                             key=lambda r: -(r.get("accuracy_ragas") or 0)):
                print(f"  {r['model_id']:<42} AC={r.get('accuracy_ragas')}  "
                      f"F={r.get('faithfulness_avg')}  AR={r.get('ar_ragas_avg')}  "
                      f"CR={r.get('context_relevance_avg')}")


if __name__ == "__main__":
    main()
