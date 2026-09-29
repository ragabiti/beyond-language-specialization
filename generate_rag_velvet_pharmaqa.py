"""
Generazione risposte RAG su PharmaQA.IT con modelli Velvet via API.

Adattato da generate_rag_velvet.py per il dominio farmaceutico italiano.
Utilizza il wrapper API Velvet (src/llm_velvet.LLM) per l'inferenza
e un prompt specifico per l'estrazione di risposte brevi da Riassunti
delle Caratteristiche del Prodotto (RCP).

Dataset supportati:
    - PharmaQA.IT (domande farmaceutiche in italiano)

Modelli supportati:
    - Modelli Velvet tramite la classe ``src/llm_velvet.LLM``

Dipendenze principali:
    - os, sys, re, json, pickle : gestione file system e serializzazione
    - argparse                  : parsing degli argomenti da riga di comando
    - tqdm                      : barra di avanzamento durante la generazione
    - src/llm_velvet.LLM : wrapper API per i modelli Velvet

Uso:
    python generate_rag_velvet_pharmaqa.py --llm_id velvet-25b-011-p10
    python generate_rag_velvet_pharmaqa.py --llm_id velvet-14b
    python generate_rag_velvet_pharmaqa.py --llm_id velvet-2b-1.5-03-23918

Output:
    data/gen_res/<model>/pharmaqa_it/top5/
"""

import os
import sys
import re
import json
import glob
import pickle
import argparse
from tqdm import tqdm

# Aggiunge la cartella src al path per importare il modulo LLM Velvet
sys.path.insert(0, "src")
from llm_velvet import LLM


# Mappa nome dataset -> percorso file JSON di default
DATASET_PATHS = {
    'pharmaqa_it': 'data/dataset/pharmaqa_dataset.json',
}

# Istruzione di sistema per il modello: risposta estrattiva farmaceutica (max 10 token)
TASK_INSTRUCTION = (
    "Sei un assistente farmaceutico esperto. Ti vengono forniti estratti di un "
    "Riassunto delle Caratteristiche del Prodotto (RCP) e una domanda. "
    "Rispondi ESTRAENDO la risposta (massimo 10 token) dal documento, "
    "basandoti ESCLUSIVAMENTE sulle informazioni presenti. "
    "Se il documento non contiene la risposta, rispondi con NO-RES."
)


def build_prompt(question: str, passages: list) -> str:
    """Costruisce il prompt RAG per domande farmaceutiche via Velvet API.

    Combina l'istruzione di sistema specifica per il dominio farmaceutico,
    i passaggi estratti dagli RCP e la domanda in un unico prompt.

    Args:
        question: la domanda farmaceutica a cui rispondere.
        passages: lista di dizionari con chiave 'text' contenente
                  il testo di ciascun passaggio recuperato dagli RCP.

    Returns:
        Stringa con il prompt completo pronto per l'inferenza del modello.
    """
    # Formattazione di ogni passaggio come "Document [N](Title: passage) <testo>"
    docs_str = "\n".join(
        f"Document [{i+1}](Title: passage) {p.get('text', '')}"
        for i, p in enumerate(passages)
    )
    return f"{TASK_INSTRUCTION}\nDocuments:\n{docs_str}\nQuestion: {question}\nAnswer:"




def clean_generated(output: str) -> str:
    """Cleaning allineato a generate_rag_pharmaqa.py per output Velvet.

    Funziona sia se il wrapper restituisce solo la risposta,
    sia se restituisce accidentalmente prompt + risposta.
    Gestisce anche token artefatti tipici di alcuni tokenizer.

    Args:
        output: testo grezzo generato dal modello, potenzialmente
                None o contenente artefatti vari.

    Returns:
        Stringa pulita contenente solo la risposta effettiva.
        Restituisce stringa vuota se l'input e' None.
    """
    # Gestione input None
    if output is None:
        return ""

    output = str(output)

    # Estrazione della risposta dopo "Answer:" se il modello ripete il prompt
    if "Answer:" in output:
        output = output[output.find("Answer:") + len("Answer:"):]

    # Gestione tag di ragionamento Qwen: estrai dopo l'ultimo </think>
    if "</think>" in output:
        output = output[output.rfind("</think>") + len("</think>"):]

    # Rimozione di eventuali tag <think> residui con contenuto
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)

    # Troncamento al primo marcatore di secondo turno o nuovo documento
    for stop in ["Question:", "Document [", "Human:", "User:", "\nAnswer:"]:
        if stop in output:
            output = output[:output.find(stop)]

    # Pulizia token artefatti tipici di alcuni tokenizer
    output = output.replace("\u0120", " ")
    output = output.replace("\u010a", "\n")
    # Compressione spazi multipli in singolo
    output = re.sub(r" +", " ", output)

    return output.strip()


def string_match(generated: str, answers: list) -> bool:
    """Verifica se la risposta generata contiene una delle risposte corrette.

    Esegue un confronto approssimativo (case-insensitive) troncando
    la risposta generata ai primi separatori noti per eliminare
    eventuale testo spurio.

    Args:
        generated: testo generato dal modello come risposta.
        answers: lista di risposte corrette ammissibili.

    Returns:
        True se almeno una risposta corretta e' contenuta nel testo
        generato (dopo normalizzazione), False altrimenti.
    """
    gen = generated.strip().lower()
    # Troncamento al primo separatore spurio (newline, riferimento documento, parentesi, domanda)
    for sep in ['\n', 'Document [', '(', 'Question:']:
        if sep in gen:
            gen = gen.split(sep)[0]
    gen = gen.strip()
    # Verifica se almeno una risposta gold e' contenuta nel testo generato
    return any(ans.lower().strip() in gen for ans in answers)


def run_dataset(llm, dataset_name: str, dataset_path: str, args, llm_folder: str):
    """Esegue la generazione RAG su un singolo dataset farmaceutico per tutti i valori di k.

    Per ogni valore di k, seleziona i top-k passaggi, costruisce il prompt,
    genera la risposta tramite Velvet API e valuta la correttezza con
    string match. I risultati vengono salvati progressivamente su disco.

    Args:
        llm: istanza del modello LLM gia' inizializzata per la generazione.
        dataset_name: nome identificativo del dataset (es. 'pharmaqa_it').
        dataset_path: percorso al file JSON contenente gli esempi.
        args: namespace argparse con parametri di configurazione
              (k_values, max_samples, max_new_tokens, save_every, overwrite, output_dir).
        llm_folder: nome della cartella di output per il modello corrente.
    """
    print(f"\n{'#'*60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'#'*60}")

    # Caricamento del dataset JSON da disco
    with open(dataset_path, encoding='utf-8') as f:
        data = json.load(f)

    # Eventuale limitazione del numero di esempi (utile per debug)
    if args.max_samples:
        data = data[:args.max_samples]

    print(f"Esempi: {len(data)}")

    # Ciclo principale su ogni valore di k (numero di passaggi nel contesto)
    for k in args.k_values:
        print(f"\n{'='*50}")
        print(f"Dataset={dataset_name}  k={k} passages")
        print(f"{'='*50}")

        # Creazione della directory di output per questo dataset e k
        save_dir = os.path.join(args.output_dir, llm_folder, dataset_name, f"top{k}")
        os.makedirs(save_dir, exist_ok=True)

        # Controllo se l'esperimento e' gia' stato completato (skip se non --overwrite)
        summary_path = os.path.join(save_dir, 'summary.json')
        if not args.overwrite and os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"  Già completato — Accuracy: {summary['accuracy']:.4f}  (usa --overwrite per rigirare)")
            continue

        # --- Resume: carica risultati parziali se presenti ---
        checkpoint_path = os.path.join(save_dir, f"results_top{k}_checkpoint.pkl")
        results = []
        done_ids = set()

        # Cerca prima il checkpoint, poi eventuali pkl finali orfani
        resume_sources = [checkpoint_path] + sorted(
            glob.glob(os.path.join(save_dir, f"results_top{k}_info_*.pkl"))
        )
        for src in resume_sources:
            if os.path.exists(src):
                with open(src, 'rb') as f:
                    results = pickle.load(f)
                done_ids = {r['example_id'] for r in results}
                print(f"  Resume: {len(results)} record gia' completati")
                break

        correct = sum(1 for r in results if r.get('is_correct', False))
        new_count = 0

        # Iterazione su ogni esempio del dataset
        for idx, example in enumerate(tqdm(data, desc=f"{dataset_name} k={k}")):
            eid = example.get('example_id', idx)
            if eid in done_ids:
                continue

            # Selezione dei top-k passaggi recuperati per questo esempio
            passages = example.get('passages', [])[:k]
            # Costruzione del prompt con istruzione farmaceutica, documenti e domanda
            prompt = build_prompt(example.get('question', ''), passages)

            # Chiamata al modello Velvet per generare la risposta
            output = llm.generate(prompt, max_new_tokens=args.max_new_tokens)
            # Gestione output come lista (alcuni wrapper restituiscono lista)
            if isinstance(output, list):
                output = output[0]

            # Pulizia dell'output grezzo del modello
            generated = clean_generated(output)

            # Valutazione della correttezza tramite string match
            is_correct = string_match(generated, example.get('answers', []))
            if is_correct:
                correct += 1

            # Accumulo dei risultati per l'esempio corrente
            results.append({
                'example_id':  eid,
                'question':    example.get('question', ''),
                'answers':     example.get('answers', []),
                'generated':   generated,
                'is_correct':  is_correct,
                'k':           k,
                'dataset':     dataset_name,
                'prompt_temp': example.get('prompt_temp', ''),
                'n_relevant':  sum(1 for p in passages if p.get('is_relevant', False)),
            })

            new_count += 1

            # Checkpoint periodico: sovrascrive un singolo file
            if new_count % args.save_every == 0:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(results, f)
                acc = correct / len(results) if results else 0.0
                print(f"  Checkpoint: {len(results)}/{len(data)} — Accuracy: {acc:.4f}")

        # --- Salvataggio finale ---
        final_pkl = os.path.join(save_dir, f"results_top{k}_info_{len(results)}.pkl")
        with open(final_pkl, 'wb') as f:
            pickle.dump(results, f)

        # Pulizia: rimuove checkpoint e vecchi pkl ridondanti
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        for old in glob.glob(os.path.join(save_dir, f"results_top{k}_info_*.pkl")):
            if old != final_pkl:
                os.remove(old)

        # Calcolo e stampa dell'accuracy finale per questo valore di k
        final_acc = correct / len(data) if data else 0.0
        print(f"\n{dataset_name} k={k} — Accuracy: {final_acc:.4f}")

        # Salvataggio del riepilogo finale in JSON
        summary = {
            'llm_id':    args.llm_id,
            'dataset':   dataset_name,
            'k':         k,
            'n_samples': len(data),
            'accuracy':  round(final_acc, 4),
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


def parse_args():
    """Analizza e restituisce gli argomenti da riga di comando.

    Returns:
        Namespace argparse con tutti i parametri di configurazione:
        llm_id, datasets, k_values, output_dir, save_every,
        max_samples, max_new_tokens, overwrite.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_id',         type=str, required=True)
    parser.add_argument('--datasets',       type=str, nargs='+', default=['pharmaqa_it'])
    parser.add_argument('--k_values',       type=int, nargs='+', default=[5])
    parser.add_argument('--output_dir',     type=str, default='data/gen_res')
    parser.add_argument('--save_every',     type=int, default=100)
    parser.add_argument('--max_samples',    type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--overwrite',      action='store_true')
    return parser.parse_args()


def main():
    """Punto di ingresso principale dello script.

    Esegue le seguenti operazioni in sequenza:
    1. Parsing degli argomenti da riga di comando.
    2. Inizializzazione del modello Velvet tramite API.
    3. Iterazione sui dataset richiesti con generazione RAG.
    """
    args = parse_args()

    print(f"Modello:        {args.llm_id}")
    print(f"Dataset:        {args.datasets}")
    print(f"k values:       {args.k_values}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"Output:         {args.output_dir}")

    # Inizializzazione del modello Velvet tramite wrapper API
    llm = LLM(args.llm_id)
    # Estrazione del nome cartella dal model_id (usa l'ultima parte dopo '/')
    llm_folder = args.llm_id.split("/")[-1] if '/' in args.llm_id else args.llm_id

    # Iterazione su tutti i dataset richiesti
    for dataset_name in args.datasets:
        if dataset_name not in DATASET_PATHS:
            print(f"ATTENZIONE: dataset '{dataset_name}' non riconosciuto. "
                  f"Disponibili: {list(DATASET_PATHS.keys())}. Skip.")
            continue
        dataset_path = DATASET_PATHS[dataset_name]
        # Controllo esistenza del file dataset prima di procedere
        if not os.path.exists(dataset_path):
            print(f"\nATTENZIONE: {dataset_path} non trovato, skip.")
            continue
        run_dataset(llm, dataset_name, dataset_path, args, llm_folder)

    print("\nGenerazione completata!")


if __name__ == "__main__":
    main()
