"""
Generazione risposte RAG con modelli Velvet su dataset inglesi.

Simula un sistema RAG reale usando i top-k passage già recuperati
da un retriever esterno. Per ogni esempio, costruisce un prompt con
i documenti recuperati e genera una risposta estrattiva.

Dataset supportati:
    - TriviaQA, Natural Questions (NQ), PopQA, BioASQ

Modelli supportati:
    - Tutti i modelli caricabili tramite la classe ``src.llm.LLM``
      (tipicamente modelli Velvet).

Dipendenze principali:
    - json, pickle     : serializzazione dati e risultati
    - argparse         : parsing degli argomenti da riga di comando
    - tqdm             : barra di avanzamento durante la generazione
    - src.llm.LLM     : wrapper unificato per l'inferenza dei modelli

Uso (singolo dataset):
    python generate_rag_eng.py \\
        --llm_id velvet-25b-011-p10 \\
        --datasets triviaqa \\
        --k_values 1 2 3 5 10 15 20 25

Uso (tutti i dataset):
    python generate_rag_eng.py \\
        --llm_id velvet-25b-011-p10 \\
        --datasets triviaqa nq popqa \\
        --k_values 1 2 3 5 10 15 20 25

Struttura output:
    data/gen_res/
    └── velvet-25b-011-p10/
        ├── triviaqa/
        │   ├── top1/
        │   │   ├── results_top1_info_100.pkl
        │   │   └── summary.json
        │   └── top5/
        ├── nq/
        └── popqa/
"""

import os
import sys
import json
import pickle
import argparse
from tqdm import tqdm

# Aggiunge la cartella src al path per importare il modulo LLM
sys.path.insert(0, "src")
from llm import LLM


# Mappa nome dataset -> percorso file JSON di default
DATASET_PATHS = {
    'triviaqa': 'data/triviaqa_dataset.json',
    'nq':       'data/nq_dataset.json',
    'popqa':    'data/popqa_dataset.json',
    'bioasq':   'data/bioasq_dataset.json',
}

# Istruzione di sistema per il modello: risposta estrattiva breve (max 5 token)
TASK_INSTRUCTION = (
    "You are given a question and you MUST respond by EXTRACTING the answer "
    "(max 5 tokens) from one of the provided documents. "
    "If none of the documents contain the answer, respond with NO-RES."
)


def build_prompt(question: str, passages: list) -> str:
    """Costruisce il prompt RAG completo con istruzione, documenti e domanda.

    Combina l'istruzione di sistema, i passaggi recuperati e la domanda
    in un unico prompt formattato per il modello.

    Args:
        question: la domanda dell'utente a cui rispondere.
        passages: lista di dizionari, ciascuno con chiave 'text'
                  contenente il testo del passaggio recuperato.

    Returns:
        Stringa con il prompt completo pronto per l'inferenza del modello.
    """
    # Formatta ogni passaggio come "Document [N](Title: passage) <testo>"
    docs_str = "\n".join(
        f"Document [{i+1}](Title: passage) {p['text']}"
        for i, p in enumerate(passages)
    )
    return f"{TASK_INSTRUCTION}\nDocuments:\n{docs_str}\nQuestion: {question}\nAnswer:"


def string_match(generated: str, answers: list) -> bool:
    """Verifica se la risposta generata contiene una delle risposte corrette.

    Esegue un confronto approssimativo (case-insensitive) troncando
    la risposta generata ai primi separatori noti per eliminare
    eventuale testo spuri prodotto dal modello.

    Args:
        generated: testo generato dal modello come risposta.
        answers: lista di risposte corrette ammissibili.

    Returns:
        True se almeno una risposta corretta e' contenuta nel testo
        generato (dopo normalizzazione), False altrimenti.
    """
    gen = generated.strip().lower()
    # Tronca la risposta al primo separatore spurio (newline, riferimento a documento, parentesi)
    for sep in ['\n', 'Document [', '(']:
        if sep in gen:
            gen = gen.split(sep)[0]
    gen = gen.strip()
    # Verifica se almeno una risposta gold e' contenuta nel testo generato
    return any(ans.lower() in gen for ans in answers)


def run_dataset(llm, dataset_name: str, dataset_path: str, args, llm_folder: str):
    """Esegue la generazione RAG su un singolo dataset per tutti i valori di k.

    Per ogni valore di k, seleziona i top-k passaggi, costruisce il prompt,
    genera la risposta tramite il modello e valuta la correttezza con
    string match. I risultati vengono salvati progressivamente su disco.

    Args:
        llm: istanza del modello LLM gia' inizializzata per la generazione.
        dataset_name: nome identificativo del dataset (es. 'triviaqa', 'nq').
        dataset_path: percorso al file JSON contenente gli esempi del dataset.
        args: namespace argparse con parametri di configurazione
              (k_values, max_samples, max_new_tokens, save_every, overwrite, output_dir).
        llm_folder: nome della cartella di output per il modello corrente.
    """
    print(f"\n{'#'*60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'#'*60}")

    # Caricamento del dataset JSON da disco
    print(f"Caricamento: {dataset_path}")
    with open(dataset_path) as f:
        data = json.load(f)

    # Eventuale limitazione del numero di esempi (utile per debug)
    if args.max_samples:
        data = data[:args.max_samples]

    print(f"Esempi: {len(data)}")

    # Ciclo principale su ogni valore di k (numero di passaggi nel contesto)
    for k in args.k_values:
        print(f"\n{'='*50}")
        print(f"Dataset={dataset_name}  k={k} passage")
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

        results = []
        correct = 0

        # Iterazione su ogni esempio del dataset
        for idx, example in enumerate(tqdm(data, desc=f"{dataset_name} k={k}")):
            # Selezione dei top-k passaggi recuperati per questo esempio
            passages = example['passages'][:k]
            # Costruzione del prompt con istruzione, documenti e domanda
            prompt   = build_prompt(example['question'], passages)

            # Chiamata al modello per generare la risposta
            output = llm.generate(prompt, max_new_tokens=args.max_new_tokens)
            # Gestione output come lista (alcuni modelli restituiscono lista)
            if isinstance(output, list):
                output = output[0]

            # Parsing dell'output: estrazione della risposta dopo "Answer:"
            if "Answer:" in output:
                start = output.find("Answer:") + len("Answer:")
                generated = output[start:].strip()
            else:
                generated = output.strip()

            # Valutazione della correttezza tramite string match
            is_correct = string_match(generated, example['answers'])
            if is_correct:
                correct += 1

            # Accumulo dei risultati per l'esempio corrente
            results.append({
                'example_id': example['example_id'],
                'question':   example['question'],
                'answers':    example['answers'],
                'generated':  generated,
                'is_correct': is_correct,
                'k':          k,
                'dataset':    dataset_name,
                'n_relevant': sum(1 for p in passages if p['is_relevant']),
            })

            # Salvataggio progressivo su disco ogni save_every esempi o alla fine
            if (idx + 1) % args.save_every == 0 or (idx + 1) == len(data):
                acc = correct / (idx + 1)
                print(f"  Salvato a {idx+1} — Accuracy: {acc:.4f}")
                fname = os.path.join(save_dir, f"results_top{k}_info_{idx+1}.pkl")
                with open(fname, 'wb') as f:
                    pickle.dump(results, f)

        # Calcolo e stampa dell'accuracy finale per questo valore di k
        final_acc = correct / len(data)
        print(f"\n{dataset_name} k={k} — Accuracy finale: {final_acc:.4f}")

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
        llm_id, datasets, data_dir, k_values, output_dir, save_every,
        max_samples, max_new_tokens, overwrite.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_id',         type=str, required=True)
    parser.add_argument('--datasets',       type=str, nargs='+', default=list(DATASET_PATHS.keys()),
                        help=f"Dataset da usare. Scegli tra: {list(DATASET_PATHS.keys())}. Default: tutti.")
    parser.add_argument('--data_dir',       type=str, default='data/dataset',
                        help="Cartella base dove cercare i dataset (default: data/)")
    parser.add_argument('--k_values',       type=int, nargs='+', default=[1, 2, 3, 5, 10, 15, 20, 25])
    parser.add_argument('--output_dir',     type=str, default='data/gen_res')
    parser.add_argument('--save_every',     type=int, default=100)
    parser.add_argument('--max_samples',    type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=15,
                        help="Token massimi generati per risposta (default: 15, aumentare a 50 per modelli verbosi)")
    parser.add_argument('--overwrite',      action='store_true',
                        help="Riesegui anche i k già completati (default: skip)")
    return parser.parse_args()


def main():
    """Punto di ingresso principale dello script.

    Esegue le seguenti operazioni in sequenza:
    1. Parsing degli argomenti da riga di comando.
    2. Validazione dei nomi dei dataset richiesti.
    3. Inizializzazione del modello LLM.
    4. Iterazione sui dataset richiesti con generazione RAG.
    """
    args = parse_args()

    # Validazione: verifica che tutti i dataset richiesti siano supportati
    for ds in args.datasets:
        if ds not in DATASET_PATHS:
            raise ValueError(f"Dataset '{ds}' non riconosciuto. Disponibili: {list(DATASET_PATHS.keys())}")

    print(f"Modello:   {args.llm_id}")
    print(f"Dataset:   {args.datasets}")
    print(f"k values:  {args.k_values}")
    print(f"Output:    {args.output_dir}")

    # Caricamento e inizializzazione del modello LLM
    llm = LLM(args.llm_id)
    # Estrazione del nome cartella dal model_id (rimuove il prefisso organizzazione)
    llm_folder = args.llm_id.split("/")[1] if '/' in args.llm_id else args.llm_id

    # Iterazione su tutti i dataset richiesti
    for dataset_name in args.datasets:
        # Costruisce il percorso del file dataset rispetto a data_dir
        default_filename = os.path.basename(DATASET_PATHS[dataset_name])
        dataset_path = os.path.join(args.data_dir, default_filename)

        # Controllo esistenza del file dataset prima di procedere
        if not os.path.exists(dataset_path):
            print(f"\nATTENZIONE: {dataset_path} non trovato, skip {dataset_name}.")
            continue

        run_dataset(llm, dataset_name, dataset_path, args, llm_folder)

    print("\nGenerazione completata!")


if __name__ == "__main__":
    main()
