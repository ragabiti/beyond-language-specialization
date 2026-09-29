"""
generate_rag_velvet_eng.py

Generazione risposte RAG per dataset inglesi con modelli Velvet.

Versione aggiornata con le accortezze usate negli script UniQA/PharmaQA:
  - prompt non limitato a "max 5 tokens", ma piu' secco e dataset-specific
  - cleaning robusto dell'output (thinking tags, preamble, secondo turno)
  - salvataggio progressivo su disco
  - accuracy solo indicativa; metrica principale: RAGAS / LLM-as-a-judge
  - max_new_tokens default basso: 120

Dataset supportati:
    - TriviaQA, Natural Questions (NQ), PopQA, BioASQ

Modelli supportati:
    - Modelli Velvet tramite la classe ``src/llm_velvet.LLM``

Dipendenze principali:
    - os, sys, re, json, pickle : gestione file system e serializzazione
    - argparse                  : parsing degli argomenti da riga di comando
    - tqdm                      : barra di avanzamento durante la generazione
    - src/llm_velvet.LLM : wrapper API per i modelli Velvet

Uso:
    python generate_rag_velvet_eng.py \\
        --llm_id velvet-14b \\
        --datasets triviaqa nq bioasq \\
        --k_values 5 \\
        --output_dir data/gen_res

Dataset attesi:
    data/triviaqa_dataset.json
    data/nq_dataset.json
    data/popqa_dataset.json
    data/bioasq_dataset.json
"""

import os
import sys
import re
import json
import pickle
import argparse
from tqdm import tqdm

# Aggiunge la cartella src al path per importare il modulo LLM Velvet
sys.path.insert(0, "src")
from llm_velvet import LLM


# Mappa nome dataset -> percorso file JSON di default
DATASET_PATHS = {
    "triviaqa": "data/dataset/triviaqa_dataset.json",
    "nq": "data/dataset/nq_dataset.json",
    "popqa": "data/dataset/popqa_dataset.json",
    "bioasq": "data/dataset/bioasq_dataset.json",
}


# Istruzione di sistema per Natural Questions: risposta breve e ancorata ai documenti
TASK_INSTRUCTION_NQ = (
    "You are given a question and retrieved documents. "
    "Answer using only information explicitly supported by the documents. "
    "Give the shortest complete answer possible. "
    "Prefer a noun phrase or one short sentence. "
    "Do not explain. Do not add background information. "
    "If the answer is not present, respond exactly with NO-RES."
)

# Istruzione di sistema per TriviaQA: solo l'entita' risposta
TASK_INSTRUCTION_TRIVIAQA = (
    "You are given a trivia question and retrieved documents. "
    "Answer using only information explicitly supported by the documents. "
    "Give only the answer entity or the shortest possible answer. "
    "Do not explain. Do not add background information. "
    "If the answer is not present, respond exactly with NO-RES."
)

# Istruzione di sistema per BioASQ: risposta biomedica breve
TASK_INSTRUCTION_BIOASQ = (
    "You are given a biomedical question and retrieved biomedical documents. "
    "Answer using only information explicitly supported by the documents. "
    "Give the shortest complete biomedical answer possible. "
    "Prefer a biomedical entity, noun phrase, or one short sentence. "
    "Do not explain. Do not add background information. "
    "If the answer is not present, respond exactly with NO-RES."
)

# Mappa dataset -> istruzione specifica (PopQA riusa quella di TriviaQA)
TASK_INSTRUCTIONS = {
    "triviaqa": TASK_INSTRUCTION_TRIVIAQA,
    "nq": TASK_INSTRUCTION_NQ,
    "popqa": TASK_INSTRUCTION_TRIVIAQA,
    "bioasq": TASK_INSTRUCTION_BIOASQ,
}


# Modelli con context window limitata (attualmente vuoto, pronto per estensioni future)
LIMITED_CONTEXT_MODELS = []
PASSAGE_MAX_TOKENS = 500  # token per passaggio (approssimazione: ~375 parole)


def _truncate_passage(text: str, max_tokens: int = PASSAGE_MAX_TOKENS) -> str:
    """Troncamento opzionale per modelli con context window limitata.

    Utilizza il rapporto empirico 1 token ~ 0.75 parole per convertire
    il limite in token a un limite in parole.

    Args:
        text: testo del passaggio da troncare.
        max_tokens: numero massimo di token consentiti (default: PASSAGE_MAX_TOKENS).

    Returns:
        Testo troncato con '...' finale se eccede il limite,
        oppure il testo originale se entro il limite.
    """
    max_words = int(max_tokens * 0.75)
    words = str(text).split()
    if len(words) <= max_words:
        return str(text)
    return " ".join(words[:max_words]) + "..."


def build_prompt(question: str, passages: list, dataset_name: str, llm_id: str = "") -> str:
    """Costruisce il prompt RAG con istruzione dataset-specific, documenti e domanda.

    Seleziona l'istruzione di sistema appropriata in base al dataset
    e applica il troncamento dei passaggi se il modello ha una context
    window limitata.

    Args:
        question: la domanda a cui rispondere.
        passages: lista di dizionari con chiave 'text' contenente
                  il testo di ciascun passaggio recuperato.
        dataset_name: nome del dataset ('triviaqa', 'nq', 'popqa', 'bioasq'),
                      usato per selezionare l'istruzione di sistema.
        llm_id: identificativo del modello, usato per determinare
                se applicare il troncamento dei passaggi (default: "").

    Returns:
        Stringa con il prompt completo pronto per l'inferenza.
    """
    # Selezione dell'istruzione in base al dataset
    instruction = TASK_INSTRUCTIONS.get(dataset_name, TASK_INSTRUCTION_NQ)

    # Verifica se il modello richiede troncamento dei passaggi
    llm_lower = llm_id.lower()
    needs_truncation = any(m in llm_lower for m in LIMITED_CONTEXT_MODELS)

    # Formattazione dei documenti con eventuale troncamento
    docs_str = "\n".join(
        f"Document [{i+1}](Title: passage) "
        f"{_truncate_passage(p.get('text', '')) if needs_truncation else p.get('text', '')}"
        for i, p in enumerate(passages)
    )
    return f"{instruction}\nDocuments:\n{docs_str}\nQuestion: {question}\nAnswer:"


def clean_generated(output: str) -> str:
    """Cleaning robusto per output Velvet.

    Gestisce risposta pura, prompt ripetuto, thinking tags,
    token artefatti e secondo turno di conversazione.

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

    # Gestione tag di ragionamento: estrai dopo l'ultimo </think>
    if "</think>" in output:
        output = output[output.rfind("</think>") + len("</think>"):]

    # Rimozione di eventuali tag <think> residui con contenuto
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)

    # Rimozione preamble comuni (tipici di modelli chat)
    preambles = [
        "Sure!", "Of course!", "Certainly!", "Here is the answer:", "The answer is:",
        "Based on the documents,", "Based on the provided documents,",
        "According to the documents,", "According to the provided documents,",
    ]
    stripped = output.strip()
    for p in preambles:
        if stripped.startswith(p):
            stripped = stripped[len(p):].strip()
            break
    output = stripped

    # Troncamento al primo marcatore di secondo turno o nuovo documento
    for stop in ["Question:", "Document [", "Documents:", "Human:", "User:", "Assistant:", "\nAnswer:"]:
        if stop in output:
            output = output[:output.find(stop)]

    # Pulizia token artefatti tipici di alcuni tokenizer
    output = output.replace("\u0120", " ")
    output = output.replace("\u010a", "\n")
    # Compressione spazi e newline multipli
    output = re.sub(r" +", " ", output)
    output = re.sub(r"\n{3,}", "\n\n", output)

    return output.strip()


def _normalize(text: str) -> str:
    """Normalizza testo per il confronto: rimuove markdown, spazi multipli, newline.

    Args:
        text: testo da normalizzare.

    Returns:
        Testo in minuscolo, senza formattazione markdown e con spazi singoli.
    """
    # Rimozione marcatori markdown (grassetto, intestazioni)
    text = re.sub(r"\*+", "", str(text))
    text = re.sub(r"#+\s", "", text)
    # Compressione spazi multipli in singolo
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def _key_phrases(text: str, min_words: int = 4) -> list[str]:
    """Estrae frasi significative di almeno min_words parole dal testo.

    Divide il testo in frasi (separando per punto e newline) e filtra
    quelle con almeno il numero minimo di parole specificato.

    Args:
        text: testo da cui estrarre le frasi chiave.
        min_words: numero minimo di parole per considerare una frase
                   significativa (default: 4).

    Returns:
        Lista di frasi normalizzate (lowercase, senza markdown)
        con almeno min_words parole.
    """
    # Suddivisione in frasi separando per punto e newline
    sentences = [s.strip() for s in str(text).replace("\n", ". ").split(".") if s.strip()]
    # Filtro e normalizzazione delle frasi significative
    return [_normalize(s) for s in sentences if len(s.split()) >= min_words]


def string_match(generated: str, answers: list) -> bool:
    """Match indicativo tra risposta generata e risposte gold.

    Non usare come metrica principale.
    La valutazione principale resta RAGAS / LLM-as-a-judge.

    Args:
        generated: testo della risposta generata dal modello.
        answers: lista delle risposte corrette (gold standard).

    Returns:
        True se almeno una risposta gold (o una sua frase chiave)
        e' contenuta nella risposta generata, False altrimenti.
    """
    gen = _normalize(generated)
    # Troncamento a separatori spurii per isolare la risposta effettiva
    for sep in ["\n", "Document [", "Question:", "Human:", "User:"]:
        if sep.lower() in gen:
            gen = gen.split(sep.lower())[0].strip()

    for ans in answers:
        # Match diretto: la risposta gold normalizzata e' contenuta nel testo
        ans_norm = _normalize(ans)
        if ans_norm and ans_norm in gen:
            return True
        # Match su frasi chiave (>=4 parole) della risposta gold
        for phrase in _key_phrases(ans):
            if phrase and phrase in gen:
                return True
    return False


def _safe_n_relevant(passages: list) -> int:
    """Conta il numero di passaggi rilevanti in modo sicuro.

    Gestisce il caso in cui il campo 'is_relevant' sia assente
    nel dizionario del passaggio.

    Args:
        passages: lista di dizionari dei passaggi, ciascuno con
                  campo opzionale 'is_relevant' (booleano).

    Returns:
        Numero intero di passaggi con is_relevant=True.
    """
    return sum(1 for p in passages if p.get("is_relevant", False))


def run_dataset(llm, dataset_name: str, dataset_path: str, args, llm_folder: str):
    """Esegue la generazione RAG su un singolo dataset inglese per tutti i valori di k.

    Per ogni valore di k, seleziona i top-k passaggi, costruisce il prompt
    dataset-specific, genera la risposta tramite Velvet API e valuta la
    correttezza con string match. I risultati vengono salvati progressivamente.

    Args:
        llm: istanza del modello LLM gia' inizializzata per la generazione.
        dataset_name: nome identificativo del dataset (es. 'triviaqa', 'nq').
        dataset_path: percorso al file JSON contenente gli esempi.
        args: namespace argparse con parametri di configurazione
              (k_values, max_samples, max_new_tokens, save_every, overwrite, output_dir, llm_id).
        llm_folder: nome della cartella di output per il modello corrente.
    """
    print(f"\n{'#'*60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'#'*60}")
    print(f"Caricamento: {dataset_path}")

    # Caricamento del dataset JSON da disco
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    # Eventuale limitazione del numero di esempi (utile per debug)
    if args.max_samples:
        data = data[:args.max_samples]

    # Segnalazione se il modello necessita di troncamento passaggi
    llm_lower = args.llm_id.lower()
    if any(m in llm_lower for m in LIMITED_CONTEXT_MODELS):
        print(
            f"  [context limit] Passaggi troncati a {PASSAGE_MAX_TOKENS} token "
            f"({int(PASSAGE_MAX_TOKENS * 0.75)} parole)"
        )

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
        summary_path = os.path.join(save_dir, "summary.json")
        if not args.overwrite and os.path.exists(summary_path):
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            print(
                f"  Già completato — Accuracy indicativa: {summary['accuracy']:.4f} "
                f"(usa --overwrite per rigirare)"
            )
            continue

        results = []
        correct = 0

        # Iterazione su ogni esempio del dataset
        for idx, example in enumerate(tqdm(data, desc=f"{dataset_name} k={k}")):
            # Selezione dei top-k passaggi recuperati per questo esempio
            passages = example["passages"][:k]
            # Costruzione del prompt dataset-specific con istruzione, documenti e domanda
            prompt = build_prompt(example["question"], passages, dataset_name, args.llm_id)

            # Chiamata al modello Velvet per generare la risposta
            output = llm.generate(prompt, max_new_tokens=args.max_new_tokens)
            # Gestione output come lista (alcuni wrapper restituiscono lista)
            if isinstance(output, list):
                output = output[0]

            # Pulizia robusta dell'output grezzo del modello
            generated = clean_generated(output)

            # Valutazione della correttezza tramite string match (metrica indicativa)
            is_correct = string_match(generated, example.get("answers", []))
            if is_correct:
                correct += 1

            # Accumulo dei risultati per l'esempio corrente
            results.append({
                "example_id": example.get("example_id", idx),
                "question": example.get("question", ""),
                "answers": example.get("answers", []),
                "generated": generated,
                "is_correct": is_correct,
                "k": k,
                "dataset": dataset_name,
                "n_relevant": _safe_n_relevant(passages),
            })

            # Salvataggio progressivo su disco ogni save_every esempi o alla fine
            if (idx + 1) % args.save_every == 0 or (idx + 1) == len(data):
                acc = correct / (idx + 1)
                print(f"  [{idx+1}/{len(data)}] Accuracy indicativa: {acc:.4f}")
                fname = os.path.join(save_dir, f"results_top{k}_info_{idx+1}.pkl")
                with open(fname, "wb") as f:
                    pickle.dump(results, f)

        # Calcolo e stampa dell'accuracy finale per questo valore di k
        final_acc = correct / len(data) if data else 0.0
        print(f"\n{dataset_name} k={k} — Accuracy indicativa: {final_acc:.4f}")
        print("  (metrica principale: RAGAS con LLM judge)")

        # Salvataggio del riepilogo finale in JSON con metadati sull'esperimento
        summary = {
            "llm_id": args.llm_id,
            "dataset": dataset_name,
            "k": k,
            "n_samples": len(data),
            "accuracy": round(final_acc, 4),
            "accuracy_note": "indicativa — metrica principale: RAGAS / LLM-as-a-judge",
            "max_new_tokens": args.max_new_tokens,
            "prompt_style": "short grounded answer, dataset-specific instruction, no max-5-token extraction",
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def parse_args():
    """Analizza e restituisce gli argomenti da riga di comando.

    Returns:
        Namespace argparse con tutti i parametri di configurazione:
        llm_id, datasets, data_dir, k_values, output_dir, save_every,
        max_samples, max_new_tokens, overwrite.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_id", type=str, required=True)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["triviaqa", "nq", "bioasq"],
        help=f"Dataset da usare. Disponibili: {list(DATASET_PATHS.keys())}",
    )
    parser.add_argument("--data_dir", type=str, default="data/dataset")
    parser.add_argument("--k_values", type=int, nargs="+", default=[5])
    parser.add_argument("--output_dir", type=str, default="data/gen_res")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    """Punto di ingresso principale dello script.

    Esegue le seguenti operazioni in sequenza:
    1. Parsing degli argomenti da riga di comando.
    2. Validazione dei nomi dei dataset richiesti.
    3. Inizializzazione del modello Velvet tramite API.
    4. Iterazione sui dataset richiesti con generazione RAG.
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

    # Inizializzazione del modello Velvet tramite wrapper API
    llm = LLM(args.llm_id)
    # Estrazione del nome cartella dal model_id (usa l'ultima parte dopo '/')
    llm_folder = args.llm_id.split("/")[-1] if "/" in args.llm_id else args.llm_id

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
