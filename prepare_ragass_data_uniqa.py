"""
prepare_ragass_data_uniqa.py

Prepara i dati per la valutazione RAGAS su UniQA IT.

Questo script costruisce il file JSON usato da run_ragass_phase1_uniqa.py.
Per ogni modello carica i risultati generati nei file .pkl e aggiunge il
contesto RAG necessario per calcolare faithfulness e context relevance.

Dataset supportati:
    - uniqa_it

Modelli supportati:
    10 modelli di generazione (vedi dizionario MODELS nel sorgente).

Dipendenze principali:
    - pickle   : deserializzazione dei risultati generati (.pkl)
    - json     : lettura del dataset UniQA e scrittura dell'output
    - random   : campionamento riproducibile con seed fisso (usato se N_MAX != None)
    - pathlib  : gestione cross-platform dei percorsi
    - argparse : parsing degli argomenti da riga di comando

Scelte metodologiche:
  - Dataset: uniqa_it.
  - Sample: tutti gli example_id comuni ai modelli disponibili, salvo N_MAX.
  - Contesto: ricostruito da UniQA/data/uniqa_it_dataset.json, non dai .pkl.
    I .pkl UniQA salvano domanda, gold answers, risposta generata e metadati,
    ma non salvano direttamente i testi dei passage recuperati.
  - Il contesto usato per RAGAS e' il top-5 ordinato per rank del dataset
    UniQA gia' arricchito con retrieval FAISS/BGE.
  - Questa scelta e' corretta solo se UniQA/data/uniqa_it_dataset.json e' lo
    stesso file usato durante la generazione delle risposte.
  - Tutti i modelli elencati sono valutati su top5.

Output:
    data/ragass/ragass_samples_uniqa_it.json

Esempio d'uso:
    python prepare_ragass_data_uniqa.py
    python prepare_ragass_data_uniqa.py --output-file data/ragass/custom_output.json
"""

import pickle
import json
import random
import argparse
from pathlib import Path

# ─── Configurazione globale ──────────────────────────────────────────────────
# Seed per la riproducibilita' del campionamento
SEED    = 42
# Limite massimo di sample; None = usa tutti gli example_id comuni disponibili
N_MAX   = None

# Percorsi relativi alla root del progetto
GEN_RES_DIR = Path("data/gen_res")
OUT_PATH    = Path("data/ragass/ragass_samples_uniqa_it.json")

# Dizionario modelli -> top_k usato per la valutazione.
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
    "velvet-14b":                           5,
    "velvet-2b-1.5-03-23918":              5,
}

# Nome del dataset UniQA italiano
DATASET = "uniqa_it"


# ─── Funzioni di supporto ────────────────────────────────────────────────────

def model_slug(model_id: str) -> str:
    """
    Converte il model_id nel nome della cartella in data/gen_res/.

    Estrae la parte dopo l'ultimo '/' dall'identificativo del modello.
    Se non contiene '/', restituisce il model_id invariato.

    Args:
        model_id: identificativo completo del modello, eventualmente con
            prefisso organizzazione separato da '/'.

    Returns:
        Stringa con il solo nome del modello (parte dopo l'ultimo '/'),
        oppure il model_id invariato se non contiene '/'.
    """
    return model_id.split("/")[-1] if "/" in model_id else model_id


def load_pkl_for_model(model_id: str, top_k: int) -> dict:
    """
    Carica tutti i file pkl di un modello sul dataset uniqa_it.

    Scandisce la directory data/gen_res/<slug>/uniqa_it/top<top_k>/
    alla ricerca di file con pattern results_top<top_k>_info_*.pkl, li
    deserializza e restituisce un dizionario indicizzato per example_id.

    Args:
        model_id: identificativo completo del modello (es. "google/gemma-4-E2B-it").
        top_k: numero di passaggi top-k usati durante la generazione.

    Returns:
        Dizionario {example_id: record} dove ogni record e' un dict con
        campi quali 'question', 'answers', 'generated', 'is_correct', ecc.

    Raises:
        FileNotFoundError: se la directory del modello/dataset non esiste.
        ValueError: se non viene trovato nessun record nella directory.
    """
    # Costruisce il percorso della directory dei risultati
    base = GEN_RES_DIR / model_slug(model_id) / DATASET / f"top{top_k}"
    # Verifica l'esistenza della directory prima di procedere
    if not base.exists():
        raise FileNotFoundError(f"Directory non trovata: {base}")

    records = {}
    # Itera su tutti i file pkl ordinati per nome nella directory
    for pkl_file in sorted(base.glob(f"results_top{top_k}_info_*.pkl")):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        # Indicizza ogni record per example_id (sovrascrive eventuali duplicati)
        for item in data:
            records[item["example_id"]] = item

    # Segnala errore se la directory esiste ma non contiene record validi
    if not records:
        raise ValueError(f"Nessun record trovato in {base}")
    return records


def build_context_from_record(rec: dict, top_k: int) -> str:
    """
    Fallback non usato nel flusso principale.

    I file .pkl di UniQA attuali non contengono i testi dei passage
    recuperati. Per questo motivo il contesto viene ricostruito dalla
    funzione load_uniqa_passages(), che legge direttamente
    UniQA/data/uniqa_it_dataset.json.

    Questa funzione esiste come fallback nel caso in cui i .pkl futuri
    includano il campo 'context' direttamente.

    Args:
        rec: dizionario di un singolo record pkl contenente i dati
            dell'esempio (question, answers, generated, ecc.).
        top_k: numero di passaggi top-k attesi (non utilizzato
            nell'implementazione corrente).

    Returns:
        Stringa del contesto estratta dal record, oppure stringa vuota
        se il campo 'context' non e' presente.
    """
    return rec.get("context", "")


def load_uniqa_passages() -> dict:
    """
    Carica i passaggi recuperati per UniQA IT dal file JSON arricchito.

    Il file UniQA/data/uniqa_it_dataset.json deve essere quello prodotto
    dopo il retrieval FAISS/BGE: ogni esempio contiene 25 passage ordinabili
    tramite il campo "rank". Per RAGAS viene usato il top-5.

    Args:
        Nessuno. Il percorso del file e' definito internamente.

    Returns:
        Dizionario {example_id: context_string} dove context_string e' una
        stringa formattata contenente i primi 5 passaggi ordinati per rank,
        ciascuno preceduto da un'etichetta "[Document N]".

    Raises:
        FileNotFoundError: se il file UniQA/data/uniqa_it_dataset.json
            non esiste su disco.
    """
    # Percorso del file del dataset UniQA arricchito con retrieval
    dataset_path = Path("UniQA/data/uniqa_it_dataset.json")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")

    # Lettura del file JSON
    with open(dataset_path, encoding="utf-8") as f:
        raw = json.load(f)

    passages_map = {}
    for ex in raw:
        eid = str(ex["example_id"])
        # Ordina i passage per rank crescente e prende i primi 5
        sorted_p = sorted(ex["passages"], key=lambda p: p["rank"])
        # Formatta i passaggi in un'unica stringa numerata per RAGAS
        context_str = "\n\n".join(
            f"[Document {i+1}] {p['text']}"
            for i, p in enumerate(sorted_p[:5])
        )
        passages_map[eid] = context_str

    return passages_map


# ─── Funzione principale ─────────────────────────────────────────────────────

def main():
    """
    Punto di ingresso principale dello script.

    Coordina l'intero flusso di preparazione dei dati RAGAS per il dataset
    UniQA IT: caricamento dei pkl, intersezione degli example_id,
    campionamento opzionale (se N_MAX != None), ricostruzione del contesto
    dai passaggi UniQA e serializzazione in JSON.

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

    print(f"\n{'='*55}")
    print(f"  Dataset: {DATASET.upper()}")
    print(f"{'='*55}")

    # --- Passo 1: carica i file pkl per ogni modello sul dataset uniqa_it ---
    model_records = {}
    for m, top_k in MODELS.items():
        try:
            recs = load_pkl_for_model(m, top_k)
            model_records[m] = recs
            print(f"  ✓ {model_slug(m):<40} {len(recs)} esempi  (top{top_k})")
        except Exception as e:
            # Se il caricamento fallisce, il modello viene ignorato
            print(f"  ✗ {model_slug(m):<40} SKIP ({e})")

    # Se nessun modello ha dati disponibili, interrompe l'esecuzione
    if not model_records:
        print("  Nessun modello disponibile.")
        return

    # --- Passo 2: calcola l'intersezione degli example_id comuni a tutti i modelli ---
    common_ids = set.intersection(
        *[set(r.keys()) for r in model_records.values()]
    )
    print(f"\n  example_id comuni a tutti i modelli: {len(common_ids)}")

    # --- Passo 3: campionamento (tutti gli id oppure limitato a N_MAX) ---
    if N_MAX is not None:
        # Se N_MAX e' impostato, campiona al massimo N_MAX example_id
        n_sample = min(N_MAX, len(common_ids))
        sampled_ids = rng.sample(sorted(common_ids), n_sample)
        print(f"  Campionati (N_MAX={N_MAX}): {len(sampled_ids)}")
    else:
        # Altrimenti usa tutti gli example_id comuni, ordinati per determinismo
        sampled_ids = sorted(common_ids)
        print(f"  Uso tutti: {len(sampled_ids)}")

    # --- Passo 4: carica i passaggi dal dataset UniQA arricchito con retrieval ---
    print(f"\n  Caricamento passaggi da UniQA/data/uniqa_it_dataset.json...")
    try:
        passages_map = load_uniqa_passages()
        print(f"  Passages caricati: {len(passages_map)} esempi")
    except FileNotFoundError as e:
        # Se il file dei passaggi non esiste, il contesto sara' vuoto
        print(f"  ⚠️  {e}")
        print(f"  Il contesto sarà vuoto — Faithfulness e CR non calcolabili")
        passages_map = {}

    # --- Passo 5: costruisce i record RAGAS per ogni coppia modello x sample ---
    output = []
    for eid in sampled_ids:
        # Recupera il contesto pre-formattato per l'esempio corrente
        context_str = passages_map.get(str(eid), "")

        # Itera su tutti i modelli caricati per creare un record per ciascuno
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

    # ── Passo 6: riepilogo finale e salvataggio ──────────────────────────
    # Calcola le statistiche aggregate sull'output prodotto
    n_models  = len(set(r["model_id"] for r in output))
    n_samples = len(set(r["example_id"] for r in output))

    print(f"\n{'='*55}")
    print(f"  Totale record:  {len(output)}")
    print(f"  Modelli:        {n_models}")
    print(f"  Sample unici:   {n_samples}")
    print(f"  Attesi totali:  {n_samples * n_models}")
    print(f"{'='*55}")

    # Serializza l'output in formato JSON con encoding UTF-8
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Salvato in: {out_path}")
    print(f"  Ora lancia: python run_ragass_phase1_uniqa.py")


if __name__ == "__main__":
    main()
