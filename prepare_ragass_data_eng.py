"""
prepare_ragass_data_eng.py

Prepara i dati per la valutazione RAGAS partendo dai file pkl già generati
per i dataset inglesi (TriviaQA, NaturalQuestions, BioASQ).

Dataset supportati:
    - triviaqa  (334 sample)
    - nq        (333 sample)
    - bioasq    (333 sample)

Modelli supportati:
    11 modelli di generazione (vedi lista MODELS nel sorgente).

Dipendenze principali:
    - pickle   : deserializzazione dei risultati generati (.pkl)
    - json     : lettura dei dataset originali e scrittura dell'output
    - random   : campionamento riproducibile con seed fisso
    - pathlib  : gestione cross-platform dei percorsi
    - argparse : parsing degli argomenti da riga di comando

Cosa fa:
  1. Carica i pkl di tutti i modelli per ogni dataset
  2. Trova gli example_id comuni a tutti i modelli
  3. Campiona N_PER_DS esempi per dataset con seed fisso (riproducibile)
  4. Ricostruisce il contesto top-k dai dataset originali
  5. Salva tutto in un unico file JSON flat per run_ragass.py

Output: data/ragass/ragass_samples_eng.json
        -> 11 modelli x 1000 sample = 11.000 record

Esempio d'uso:
    python prepare_ragass_data_eng.py
    python prepare_ragass_data_eng.py --output-file data/ragass/custom_output.json
"""

import pickle
import json
import random
import argparse
from pathlib import Path

# ─── Configurazione globale ──────────────────────────────────────────────────
# Seed per la riproducibilità del campionamento
SEED  = 42
# Numero di passaggi di contesto da utilizzare per ogni esempio
TOP_K = 5

# Numero di sample da campionare per ciascun dataset (334 + 333 + 333 = 1000)
N_PER_DS = {
    "triviaqa": 334,
    "nq":       333,
    "bioasq":   333,
}

# Percorsi relativi alla root del progetto
GEN_RES_DIR = Path("data/gen_res")
DATASET_DIR = Path("data/dataset")
OUT_PATH    = Path("data/ragass/ragass_samples_eng.json")

# Lista dei dataset inglesi da processare
DATASETS = ["triviaqa", "nq", "bioasq"]

# Lista completa dei modelli da includere nella valutazione
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


# ─── Funzioni di supporto ────────────────────────────────────────────────────
def model_slug(model_id: str) -> str:
    """
    Converte il model_id nel nome della cartella in data/gen_res/.

    Le cartelle usano solo il nome del modello senza il prefisso organizzazione.
    Esempi:
        google/gemma-4-E2B-it       -> gemma-4-E2B-it
        ibm-granite/granite-4.1-3B  -> granite-4.1-3B
        Qwen/Qwen3.5-9B            -> Qwen3.5-9B
        velvet-25b-011-p10          -> velvet-25b-011-p10

    Args:
        model_id: identificativo completo del modello, eventualmente con
            prefisso organizzazione separato da '/'.

    Returns:
        Stringa con il solo nome del modello (parte dopo l'ultimo '/'),
        oppure il model_id invariato se non contiene '/'.
    """
    return model_id.split("/")[-1] if "/" in model_id else model_id


def load_pkl_for_model_dataset(model_id: str, dataset: str) -> dict:
    """
    Carica tutti i file pkl di un modello per un dato dataset.

    Scandisce la directory data/gen_res/<slug>/<dataset>/top<TOP_K>/ alla
    ricerca di file con pattern results_top<TOP_K>_info_*.pkl, li
    deserializza e restituisce un dizionario indicizzato per example_id.

    Args:
        model_id: identificativo completo del modello (es. "google/gemma-4-E2B-it").
        dataset: nome del dataset (es. "triviaqa", "nq", "bioasq").

    Returns:
        Dizionario {example_id: record} dove ogni record e' un dict con
        campi quali 'question', 'answers', 'generated', 'is_correct', ecc.

    Raises:
        FileNotFoundError: se la directory del modello/dataset non esiste.
        ValueError: se non viene trovato nessun record nella directory.
    """
    # Costruisce il percorso della directory dei risultati
    base = GEN_RES_DIR / model_slug(model_id) / dataset / f"top{TOP_K}"
    # Verifica l'esistenza della directory prima di procedere
    if not base.exists():
        raise FileNotFoundError(f"Directory non trovata: {base}")

    records = {}
    # Itera su tutti i file pkl ordinati per nome nella directory
    for pkl_file in sorted(base.glob(f"results_top{TOP_K}_info_*.pkl")):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        # Indicizza ogni record per example_id (sovrascrive eventuali duplicati)
        for item in data:
            records[item["example_id"]] = item

    # Segnala errore se la directory esiste ma non contiene record validi
    if not records:
        raise ValueError(f"Nessun record trovato in {base}")
    return records


def load_passages(dataset: str) -> dict:
    """
    Carica i passaggi originali dal file JSON del dataset.

    Cerca il file del dataset in due possibili posizioni:
        1. data/dataset/<dataset>_dataset.json
        2. data/dataset/<dataset>.json
    Per ciascun esempio, ordina i passaggi per rank e restituisce i
    primi TOP_K testi.

    Args:
        dataset: nome del dataset (es. "triviaqa", "nq", "bioasq").

    Returns:
        Dizionario {example_id: lista_di_testi} dove lista_di_testi
        contiene fino a TOP_K stringhe di passaggi ordinati per rank.

    Raises:
        FileNotFoundError: se nessuno dei percorsi candidati esiste su disco.
    """
    # Percorsi candidati in ordine di priorita'
    candidates = [
        DATASET_DIR / f"{dataset}_dataset.json",
        DATASET_DIR / f"{dataset}.json",
    ]
    # Seleziona il primo percorso esistente
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Dataset non trovato. Cercato: {[str(c) for c in candidates]}"
        )

    # Lettura del file JSON del dataset
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    passages = {}
    for ex in raw:
        eid = str(ex["example_id"])
        # Ordina i passaggi per rank crescente e prende i primi TOP_K
        sorted_p = sorted(ex["passages"], key=lambda p: p["rank"])
        passages[eid] = [p["text"] for p in sorted_p[:TOP_K]]
    return passages


# ─── Funzione principale ─────────────────────────────────────────────────────
def main():
    """
    Punto di ingresso principale dello script.

    Coordina l'intero flusso di preparazione dei dati RAGAS per i dataset
    inglesi: caricamento dei pkl, intersezione degli example_id, campionamento
    riproducibile, ricostruzione del contesto e serializzazione in JSON.

    Il file di output puo' essere personalizzato tramite l'argomento
    --output-file da riga di comando.

    Returns:
        None. Lo script produce come side-effect il file JSON di output.
    """
    # Parsing degli argomenti da riga di comando
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=str, default=None,
                        help=f"Path del file JSON di output (default: {OUT_PATH})")
    args = parser.parse_args()

    # Determina il percorso di output e crea la directory se necessario
    out_path = Path(args.output_file) if args.output_file else OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Inizializza il generatore di numeri casuali con seed fisso per riproducibilita'
    rng = random.Random(SEED)
    output = []

    # Ciclo principale su ciascun dataset
    for dataset in DATASETS:
        n_target = N_PER_DS[dataset]
        print(f"\n{'='*55}")
        print(f"  Dataset: {dataset.upper()}  (target: {n_target} sample)")
        print(f"{'='*55}")

        # --- Passo 1: carica i file pkl per ogni modello sul dataset corrente ---
        model_records = {}
        for m in MODELS:
            try:
                recs = load_pkl_for_model_dataset(m, dataset)
                model_records[m] = recs
                print(f"  ✓ {m:<45} {len(recs)} esempi")
            except Exception as e:
                # Se il caricamento fallisce, il modello viene ignorato per questo dataset
                print(f"  ✗ {m:<45} SKIP ({e})")

        # Se nessun modello ha dati per questo dataset, salta al prossimo
        if not model_records:
            print("  Nessun modello disponibile, skip dataset.")
            continue

        # --- Passo 2: calcola l'intersezione degli example_id comuni a tutti i modelli ---
        common_ids = set.intersection(
            *[set(r.keys()) for r in model_records.values()]
        )
        print(f"\n  example_id comuni: {len(common_ids)}")

        # Adatta il numero di sample se gli id comuni sono meno del target
        n_sample = min(n_target, len(common_ids))
        if n_sample < n_target:
            print(f"  ⚠️  Meno di {n_target} esempi comuni, uso {n_sample}")

        # --- Passo 3: campionamento riproducibile (sorted per determinismo) ---
        sampled_ids = rng.sample(sorted(common_ids), n_sample)
        print(f"  Campionati: {len(sampled_ids)}")

        # --- Passo 4: carica i passaggi originali dal file JSON del dataset ---
        try:
            passages_map = load_passages(dataset)
            print(f"  Passages caricati: {len(passages_map)} esempi")
        except FileNotFoundError as e:
            # Se il file dei passaggi non esiste, il contesto sara' vuoto
            print(f"  ⚠️  {e}")
            print(f"  Il contesto sarà vuoto (Faithfulness non calcolabile)")
            passages_map = {}

        # --- Passo 5: costruisce un record RAGAS per ogni coppia modello x sample ---
        for eid in sampled_ids:
            # Recupera i passaggi di contesto per l'esempio corrente
            context_passages = passages_map.get(str(eid), [])
            # Formatta i passaggi in un'unica stringa numerata
            context_str = "\n\n".join(
                f"[Document {i+1}] {p}"
                for i, p in enumerate(context_passages)
            )

            # Itera su tutti i modelli caricati per creare un record per ciascuno
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

    # ── Passo 6: riepilogo finale e salvataggio ──────────────────────────
    # Calcola le statistiche aggregate sull'output prodotto
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

    # Serializza l'output in formato JSON con encoding UTF-8
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Salvato in: {out_path}")


if __name__ == "__main__":
    main()
