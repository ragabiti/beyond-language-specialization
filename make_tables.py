"""
make_tables.py — Genera a terminale le tabelle del paper a partire
dai file JSON di summary presenti in results/.

Tabelle prodotte (per ciascun judge):
  - english  : confronto modelli su triviaqa, nq, bioasq
  - uniqa    : confronto modelli su uniqa_it
  - pharmaqa : confronto modelli su pharmaqa_it

I file di input vengono cercati in results/<judge>/<gruppo>.json,
dove <judge> e una delle directory definite in JUDGES e <gruppo>
e uno dei valori in GROUPS.

Per ogni combinazione judge/gruppo/dataset viene stampata una tabella
con le colonne: model, AC (accuracy), F (faithfulness),
AR (answer relevance), CR (context relevance).

Dipendenze:
  - json    : lettura dei file JSON di summary
  - pathlib : gestione multipiattaforma dei percorsi

Uso:
    python3 make_tables.py
"""
import json
from pathlib import Path

# Dizionario dei judge supportati: chiave = nome cartella, valore = etichetta di stampa
JUDGES = {"gpt4o_judge": "GPT-4o-mini (primary)", "qwen_judge": "Qwen3-8B (secondary)"}

# Gruppi di dataset da stampare, nell'ordine desiderato
GROUPS = ["english", "uniqa", "pharmaqa"]

def fmt(x):
    """Formatta un valore numerico a 3 decimali per la stampa in tabella.

    Se il valore non e un numero (None o tipo non numerico), restituisce
    un trattino centrato per indicare dato mancante.

    Args:
        x: Valore da formattare (int, float oppure None).

    Returns:
        Stringa formattata: '0.xxx' se numerico, '  ---  ' altrimenti.
    """
    return f"{x:.3f}" if isinstance(x, (int, float)) else "  —  "

def ac_of(r):
    """Estrae il valore di accuracy dal record, con aliasing tra judge.

    I due collettori (collect_gpt.py e collect_summaries.py) usano
    nomi di campo diversi per la stessa metrica:
      - GPT-4o  : 'accuracy'
      - Qwen3   : 'accuracy_ragas'

    Args:
        r: Dizionario del record summary.

    Returns:
        Valore numerico dell'accuracy, oppure None se assente.
    """
    return r.get("accuracy", r.get("accuracy_ragas"))

def ar_of(r):
    """Estrae il valore di answer relevance dal record, con aliasing tra judge.

    Nomi di campo supportati:
      - GPT-4o  : 'answer_relevance'
      - Qwen3   : 'ar_ragas_avg'

    Args:
        r: Dizionario del record summary.

    Returns:
        Valore numerico dell'answer relevance, oppure None se assente.
    """
    return r.get("answer_relevance", r.get("ar_ragas_avg"))

def f_of(r):
    """Estrae il valore di faithfulness dal record, con aliasing tra judge.

    Nomi di campo supportati:
      - GPT-4o  : 'faithfulness'
      - Qwen3   : 'faithfulness_avg'

    Args:
        r: Dizionario del record summary.

    Returns:
        Valore numerico della faithfulness, oppure None se assente.
    """
    return r.get("faithfulness", r.get("faithfulness_avg"))

def cr_of(r):
    """Estrae il valore di context relevance dal record, con aliasing tra judge.

    Nomi di campo supportati:
      - GPT-4o  : 'context_relevance'
      - Qwen3   : 'context_relevance_avg'

    Args:
        r: Dizionario del record summary.

    Returns:
        Valore numerico della context relevance, oppure None se assente.
    """
    return r.get("context_relevance", r.get("context_relevance_avg"))

# === Ciclo principale: iterazione su ciascun judge disponibile ===
for judge, label in JUDGES.items():
    # Costruzione del percorso base per il judge corrente
    base = Path("results") / judge

    # Se la cartella del judge non esiste, lo salta
    if not base.exists():
        continue

    # Intestazione del blocco per il judge corrente
    print(f"\n{'='*72}\n  JUDGE: {label}\n{'='*72}")

    # Iterazione sui gruppi di dataset nell'ordine definito
    for g in GROUPS:
        # Percorso del file JSON di summary per il gruppo corrente
        f = base / f"{g}.json"

        # Se il file del gruppo non esiste, lo salta
        if not f.exists():
            continue

        # Lettura e parsing del file JSON con i summary
        rows = json.loads(f.read_text(encoding="utf-8"))

        # Iterazione sui dataset presenti nel gruppo (ordinati alfabeticamente)
        for ds in sorted({r["dataset"] for r in rows}):
            # Intestazione della sotto-tabella per il dataset corrente
            print(f"\n-- {g} / {ds} --")
            print(f"{'model':<42}{'AC':>8}{'F':>8}{'AR':>8}{'CR':>8}")

            # Stampa delle righe ordinate per accuracy decrescente
            for r in sorted((r for r in rows if r["dataset"] == ds),
                            key=lambda r: -(ac_of(r) or 0)):
                print(f"{r['model_id']:<42}{fmt(ac_of(r)):>8}{fmt(f_of(r)):>8}"
                      f"{fmt(ar_of(r)):>8}{fmt(cr_of(r)):>8}")
