"""
generate_rag_uniqa_vllm.py -- Generazione RAG su UniQA via vLLM per tutti i modelli.

Esegue la generazione di risposte RAG sul dataset UniQA (italiano e inglese)
utilizzando modelli serviti tramite vLLM. Supporta sia modelli Velvet (via API
proprietaria) sia modelli open-source (via endpoint OpenAI-compatible di vLLM).

Modelli supportati:
    - Velvet (API) tramite src/llm_velvet_uniqa.py
    - Tutti gli altri tramite src/llm_vllm_chat.py, che gestisce automaticamente:
        * Chat template  -> /v1/chat/completions  (gemma, granite, ministral, fastweb, villanova)
        * Raw completion -> /v1/completions        (llama, qwen, altri)
        * Thinking mode  -> soppresso per Qwen

Nessun crop -- contesto passato intero a vLLM.

Dipendenze principali:
    - os, sys, re, json, pickle : gestione file system e serializzazione
    - argparse                  : parsing degli argomenti da riga di comando
    - tqdm                      : barra di avanzamento durante la generazione
    - src/llm_vllm_chat.py     : wrapper vLLM per modelli open-source
    - src/llm_velvet_uniqa.py  : wrapper API per modelli Velvet

Prerequisito:
    vllm serve <model_id> \\
        --port 8000 --dtype bfloat16 \\
        --gpu-memory-utilization 0.92 \\
        --max-model-len 32768 \\
        --max-num-seqs 8 \\
        --disable-log-requests

Uso:
    python generate_rag_uniqa_vllm.py --llm_id google/gemma-4-E2B-it
    python generate_rag_uniqa_vllm.py --llm_id Qwen/Qwen3.5-9B
    python generate_rag_uniqa_vllm.py --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import os
import re
import sys
import json
import glob
import pickle
import argparse
from tqdm import tqdm


def _load_llm_class(model_id: str):
    """Carica la classe LLM appropriata in base al tipo di modello.

    Seleziona dinamicamente il wrapper corretto:
      - Modelli Velvet -> src/llm_velvet_uniqa.py (API proprietaria)
      - Tutti gli altri -> src/llm_vllm_chat.py (gestisce chat e raw completion via vLLM)

    Args:
        model_id: identificativo del modello (es. 'google/gemma-4-E2B-it',
                  'velvet-14b'). Se contiene 'velvet' (case-insensitive)
                  viene usato il wrapper Velvet.

    Returns:
        Classe LLM (non istanziata) pronta per essere inizializzata
        con il model_id.
    """
    sys.path.insert(0, "src")
    # Selezione del wrapper in base alla presenza di 'velvet' nel nome modello
    if "velvet" in model_id.lower():
        from llm_velvet_uniqa import LLM
    else:
        from llm_vllm_chat import LLM
    return LLM


# Mappa nome dataset -> percorso file JSON di default
DATASET_PATHS = {
    'uniqa_it': 'data/dataset/uniqa_it_dataset.json',
    'uniqa_en': 'data/dataset/uniqa_it_dataset.json',
}

# Modelli con context window limitata a 4096 token.
# I passaggi vengono troncati per evitare overflow -> generazione lenta e degenere.
# Nota metodologica: limitazione del modello, dichiarata nel paper.
LIMITED_CONTEXT_MODELS = [ ]
PASSAGE_MAX_TOKENS = 500  # token per passaggio (approssimazione: ~375 parole)


def _truncate_passage(text: str, max_tokens: int = PASSAGE_MAX_TOKENS) -> str:
    """Tronca un passaggio a un numero massimo di token (approssimazione via word count).

    Utilizza il rapporto empirico 1 token ~ 0.75 parole per l'italiano
    per convertire il limite in token a un limite in parole.

    Args:
        text: testo del passaggio da troncare.
        max_tokens: numero massimo di token consentiti (default: PASSAGE_MAX_TOKENS).

    Returns:
        Testo troncato con '...' finale se eccede il limite,
        oppure il testo originale se entro il limite.
    """
    # 1 token ~ 0.75 parole per italiano
    max_words = int(max_tokens * 0.75)
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + '...'

# ── Prompt ────────────────────────────────────────────────────────────────────
# Differenza chiave rispetto a generate_rag_eng.py:
# - niente "max 5 tokens"
# - richiede risposta completa e articolata
# - specifica italiano per uniqa_it

# Istruzione di sistema in italiano per il dataset UniQA IT
TASK_INSTRUCTION_IT = (
    "Sei un assistente universitario. Ti viene fornita una domanda e una serie di documenti. "
    "Rispondi alla domanda IN ITALIANO in modo completo e preciso, "
    "basandoti ESCLUSIVAMENTE sulle informazioni presenti nei documenti forniti. "
    "Se i documenti non contengono informazioni sufficienti, rispondi con NO-RES."
)

# Istruzione di sistema in inglese per il dataset UniQA EN
TASK_INSTRUCTION_EN = (
    "You are a university assistant. You are given a question and a series of documents. "
    "Answer the question in a complete and accurate way, "
    "based EXCLUSIVELY on the information in the provided documents. "
    "If the documents do not contain sufficient information, respond with NO-RES."
)

# Mappa dataset -> istruzione appropriata
TASK_INSTRUCTIONS = {
    'uniqa_it': TASK_INSTRUCTION_IT,
    'uniqa_en': TASK_INSTRUCTION_EN,
}


def build_prompt(question: str, passages: list, dataset_name: str, llm_id: str = "") -> str:
    """Costruisce il prompt RAG con istruzione, documenti e domanda.

    Seleziona l'istruzione di sistema appropriata in base al dataset
    (italiano o inglese) e applica il troncamento dei passaggi se il
    modello ha una context window limitata.

    Args:
        question: la domanda dell'utente a cui rispondere.
        passages: lista di dizionari con chiave 'text' contenente
                  il testo di ciascun passaggio recuperato.
        dataset_name: nome del dataset ('uniqa_it' o 'uniqa_en'),
                      usato per selezionare l'istruzione di sistema.
        llm_id: identificativo del modello, usato per determinare
                se applicare il troncamento dei passaggi (default: "").

    Returns:
        Stringa con il prompt completo pronto per l'inferenza.
    """
    # Selezione dell'istruzione in base alla lingua del dataset
    instruction = TASK_INSTRUCTIONS.get(dataset_name, TASK_INSTRUCTION_IT)

    # Verifica se il modello richiede troncamento dei passaggi
    llm_lower = llm_id.lower()
    needs_truncation = any(m in llm_lower for m in LIMITED_CONTEXT_MODELS)

    # Formattazione dei documenti con eventuale troncamento
    docs_str = "\n".join(
        f"Document [{i+1}](Title: passage) {_truncate_passage(p.get('text', '')) if needs_truncation else p.get('text', '')}"
        for i, p in enumerate(passages)
    )
    return f"{instruction}\nDocuments:\n{docs_str}\nQuestion: {question}\nAnswer:"


# ── Pulizia output modello ────────────────────────────────────────────────────

def clean_generated(output: str) -> str:
    """Pulisce l'output grezzo del modello in modo robusto per tutti i modelli.

    Gestisce:
      - Qwen3.5 / Qwen3-14B : <think>...</think> prima della risposta
      - Llama, Gemma, altri  : preamble tipo "Certo!", "Sure!", ecc.
      - Tutti                : ripetizione del prompt, secondo turno

    Args:
        output: testo grezzo generato dal modello, potenzialmente
                contenente artefatti, tag di ragionamento o ripetizioni.

    Returns:
        Stringa pulita contenente solo la risposta effettiva del modello.
    """
    # 1. Estrai la parte dopo "Answer:" se il modello ripete il prompt
    if "Answer:" in output:
        output = output[output.find("Answer:") + len("Answer:"):]

    # 2. Rimuovi thinking tags (Qwen3.5, Qwen3-14B thinking mode)
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)

    # 3. Rimuovi preamble comuni (Gemma, alcuni Llama)
    preambles = [
        "Certo!", "Certamente!", "Ecco la risposta:", "Ecco:",
        "Sure!", "Of course!", "Certainly!",
        "Based on the documents,", "Based on the provided documents,",
        "In base ai documenti forniti,",
    ]
    stripped = output.strip()
    for p in preambles:
        if stripped.startswith(p):
            stripped = stripped[len(p):]
            break
    output = stripped

    # 4. Taglia se il modello inizia un secondo turno di conversazione
    for stop in ["Question:", "Document [1]", "Human:", "User:", "\nAnswer:"]:
        if stop in output:
            output = output[:output.find(stop)]

    return output.strip()


# ── String match ──────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Normalizza testo per il confronto: rimuove markdown, spazi multipli, newline.

    Args:
        text: testo da normalizzare.

    Returns:
        Testo in minuscolo, senza formattazione markdown e con spazi singoli.
    """
    # Rimozione marcatori markdown (grassetto, intestazioni)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s', '', text)
    # Compressione spazi multipli in singolo
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def _key_phrases(text: str, min_words: int = 5) -> list[str]:
    """Estrae frasi significative di almeno min_words parole dal testo.

    Divide il testo in frasi (separando per punto e newline) e filtra
    quelle con almeno il numero minimo di parole specificato.

    Args:
        text: testo da cui estrarre le frasi chiave.
        min_words: numero minimo di parole per considerare una frase
                   significativa (default: 5).

    Returns:
        Lista di frasi normalizzate (lowercase, senza markdown)
        con almeno min_words parole.
    """
    # Suddivisione in frasi separando per punto e newline
    sentences = [s.strip() for s in text.replace('\n', '. ').split('.') if s.strip()]
    # Filtro e normalizzazione delle frasi significative
    return [_normalize(s) for s in sentences if len(s.split()) >= min_words]


def string_match(generated: str, answers: list) -> bool:
    """Match unidirezionale su frasi chiave tra risposta generata e gold.

    Verifica se la risposta generata contiene almeno una delle risposte
    corrette, sia tramite match diretto sia tramite frasi chiave.
    NOTA: metrica secondaria su UniQA -- la metrica principale e' RAGAS.

    Args:
        generated: testo della risposta generata dal modello.
        answers: lista delle risposte corrette (gold standard).

    Returns:
        True se almeno una risposta gold (o una sua frase chiave)
        e' contenuta nella risposta generata, False altrimenti.
    """
    gen = _normalize(generated)

    for ans in answers:
        # 1. Match diretto: la risposta gold normalizzata e' contenuta nel testo
        if _normalize(ans) in gen:
            return True
        # 2. Match su frasi chiave (>=5 parole) della risposta gold
        for phrase in _key_phrases(ans):
            if phrase in gen:
                return True

    return False


# ── Run per dataset ───────────────────────────────────────────────────────────

def run_dataset(llm, dataset_name: str, dataset_path: str, args, llm_folder: str):
    """Esegue la generazione RAG su un singolo dataset per tutti i valori di k.

    Per ogni valore di k, seleziona i top-k passaggi, costruisce il prompt,
    genera la risposta e valuta la correttezza. I risultati vengono salvati
    progressivamente su disco in formato pickle.

    Args:
        llm: istanza del modello LLM gia' inizializzata per la generazione.
        dataset_name: nome identificativo del dataset (es. 'uniqa_it').
        dataset_path: percorso al file JSON contenente gli esempi.
        args: namespace argparse con parametri di configurazione
              (k_values, max_samples, max_new_tokens, save_every, overwrite, output_dir, llm_id).
        llm_folder: nome della cartella di output per il modello corrente.
    """
    print(f"\n{'#'*60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'#'*60}")

    # Caricamento del dataset JSON da disco
    print(f"Caricamento: {dataset_path}")
    with open(dataset_path, encoding='utf-8') as f:
        data = json.load(f)

    # Eventuale limitazione del numero di esempi (utile per debug)
    if args.max_samples:
        data = data[:args.max_samples]

    # Segnalazione se il modello necessita di troncamento passaggi
    llm_lower = args.llm_id.lower()
    if any(m in llm_lower for m in LIMITED_CONTEXT_MODELS):
        print(f"  [context limit] Passaggi troncati a {PASSAGE_MAX_TOKENS} token ({int(PASSAGE_MAX_TOKENS*0.75)} parole) — limitazione del modello")

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
            # Costruzione del prompt con istruzione, documenti e domanda
            prompt = build_prompt(example.get('question', ''), passages, dataset_name, args.llm_id)

            # Chiamata al modello per generare la risposta
            output = llm.generate(prompt, max_new_tokens=args.max_new_tokens)
            # Gestione output come lista (alcuni modelli restituiscono lista)
            if isinstance(output, list):
                output = output[0]

            # Salvataggio output grezzo prima della pulizia (per eventuale ri-cleaning offline)
            raw_output = output if isinstance(output, str) else str(output)
            # Pulizia robusta dell'output grezzo del modello
            generated = clean_generated(output)

            # Valutazione della correttezza tramite string match
            is_correct = string_match(generated, example.get('answers', []))
            if is_correct:
                correct += 1

            # Accumulo dei risultati per l'esempio corrente
            results.append({
                'example_id': eid,
                'question':   example.get('question', ''),
                'answers':    example.get('answers', []),
                'raw_output': raw_output,
                'generated':  generated,
                'is_correct': is_correct,
                'k':          k,
                'dataset':    dataset_name,
                'n_relevant': sum(1 for p in passages if p.get('is_relevant', False)),
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
        print(f"\n{dataset_name} k={k} — Accuracy indicativa: {final_acc:.4f}")
        print(f"  (metrica principale: RAGAS con LLM judge)")

        # Salvataggio del riepilogo finale in JSON
        summary = {
            'llm_id':        args.llm_id,
            'dataset':       dataset_name,
            'k':             k,
            'n_samples':     len(data),
            'accuracy':      round(final_acc, 4),
            'accuracy_note': 'indicativa — metrica principale e RAGAS',
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


# ── Argomenti ─────────────────────────────────────────────────────────────────

def parse_args():
    """Analizza e restituisce gli argomenti da riga di comando.

    Returns:
        Namespace argparse con tutti i parametri di configurazione:
        llm_id, datasets, data_dir, k_values, output_dir, save_every,
        max_samples, max_new_tokens, overwrite.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_id',         type=str, required=True)
    parser.add_argument('--datasets',       type=str, nargs='+',
                        default=['uniqa_it'],
                        help=f"Dataset da usare. Disponibili: {list(DATASET_PATHS.keys())}")
    parser.add_argument('--data_dir',       type=str, default='data')
    parser.add_argument('--k_values',       type=int, nargs='+', default=[5])
    parser.add_argument('--output_dir',     type=str, default='data/gen_res')
    parser.add_argument('--save_every',     type=int, default=100)
    parser.add_argument('--max_samples',    type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=300,
                        help="Token massimi generati per risposta (default: 300)")
    parser.add_argument('--overwrite',      action='store_true')
    return parser.parse_args()


def main():
    """Punto di ingresso principale dello script.

    Esegue le seguenti operazioni in sequenza:
    1. Parsing degli argomenti da riga di comando.
    2. Validazione dei nomi dei dataset richiesti.
    3. Caricamento dinamico della classe LLM appropriata.
    4. Inizializzazione del modello.
    5. Iterazione sui dataset richiesti con generazione RAG.
    """
    args = parse_args()

    # Validazione: verifica che tutti i dataset richiesti siano supportati
    for ds in args.datasets:
        if ds not in DATASET_PATHS:
            raise ValueError(f"Dataset '{ds}' non riconosciuto. Disponibili: {list(DATASET_PATHS.keys())}")

    print(f"Modello:        {args.llm_id}")
    print(f"Dataset:        {args.datasets}")
    print(f"k values:       {args.k_values}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"Output:         {args.output_dir}")

    # Caricamento dinamico della classe LLM in base al tipo di modello
    LLM = _load_llm_class(args.llm_id)
    # Inizializzazione del modello
    llm = LLM(args.llm_id)
    # Estrazione del nome cartella dal model_id (rimuove il prefisso organizzazione)
    llm_folder = args.llm_id.split("/")[1] if '/' in args.llm_id else args.llm_id

    # Iterazione su tutti i dataset richiesti
    for dataset_name in args.datasets:
        dataset_path = DATASET_PATHS[dataset_name]
        # Controllo esistenza del file dataset prima di procedere
        if not os.path.exists(dataset_path):
            print(f"\nATTENZIONE: {dataset_path} non trovato, skip {dataset_name}.")
            continue
        run_dataset(llm, dataset_name, dataset_path, args, llm_folder)

    print("\nGenerazione completata!")


if __name__ == "__main__":
    main()
