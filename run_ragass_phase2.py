"""
run_ragass_phase2.py  —  Fase 2: calcolo embedding e salvataggio risultati finali

Completa la pipeline di valutazione RAGAS calcolando le metriche basate su
embedding a partire dai risultati intermedi della Fase 1 (chiamate LLM).
Produce due versioni della metrica Answer Relevance e salva i risultati
finali raggruppati per modello e dataset.

Metriche calcolate:
    AR_ragas:  (1/n) * sum_i cosine_sim(embed(q), embed(q_i))
               dove q_i sono le domande sintetiche generate in Phase 1
               -- fedele al paper Es et al. (2023)

    AR_direct: cosine_sim(embed(question), embed(generated_answer))
               -- versione semplificata, non dipende dal judge LLM
               -- piu' robusta quando il judge ha problemi con question generation

Entrambe vengono salvate nei risultati finali insieme alle metriche della Fase 1.

Dipendenze principali:
    - sentence_transformers (SentenceTransformer): modello di embedding BAAI/bge-m3
    - json: serializzazione/deserializzazione dei risultati
    - collections.defaultdict: raggruppamento dei record per modello/dataset

Prerequisito:
    Con vLLM spento, bge-m3 usa tutta la GPU disponibile.

Output: data/ragass/results/<model>/<dataset>/ragass_results.json

Uso:
    # Prima spegni vLLM (Ctrl+C nel terminale vLLM)
    python run_ragass_phase2.py                          # GPU (default)
    python run_ragass_phase2.py --cpu                    # forza CPU
    python run_ragass_phase2.py --input-file path.json   # file intermedio custom
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer

# ─── Configurazione globale ───────────────────────────────────────────────────
EMBED_MODEL          = "BAAI/bge-m3"                            # Modello di embedding multilingue
DEFAULT_INTERIM_PATH = Path("data/ragass/interim_llm_results.json")  # File intermedio della Fase 1
DEFAULT_OUT_DIR      = Path("data/ragass/results")              # Cartella di output predefinita


# ─────────────────────────────────────────────────────────────────────────────
# Embedding: calcola AR_ragas e AR_direct in un unico batch
# ─────────────────────────────────────────────────────────────────────────────
def compute_embeddings(records: list, embedder: SentenceTransformer) -> list:
    """Calcola AR_ragas e AR_direct via embedding in un unico batch.

    Args:
        records: lista di record con campi question, generated, synthetic_questions.
        embedder: modello SentenceTransformer gia' caricato.

    Returns:
        La stessa lista con campi answer_relevance_ragas e answer_relevance_direct aggiunti.
    """
    # Raccolta di tutti i testi in un unico vettore per batch encoding
    print("  Raccolta testi...")
    all_texts = []
    index_map = []  # (indice_record, tipo) per riassegnare gli embedding dopo

    for i, rec in enumerate(records):
        all_texts.append(rec["question"])
        index_map.append((i, "question"))

        all_texts.append(rec.get("generated", ""))
        index_map.append((i, "generated"))

        for q in rec.get("synthetic_questions", []):
            if q and q.strip():
                all_texts.append(q)
                index_map.append((i, "synth"))

    print(f"  Testi da embeddare: {len(all_texts)}")
    embeddings = embedder.encode(
        all_texts,
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=True,
    )

    # Riorganizza gli embedding per record e tipo
    question_embs = {}
    generated_embs = {}
    synth_embs = defaultdict(list)

    for (idx, tipo), emb in zip(index_map, embeddings):
        if tipo == "question":
            question_embs[idx] = emb
        elif tipo == "generated":
            generated_embs[idx] = emb
        elif tipo == "synth":
            synth_embs[idx].append(emb)

    # Calcola le due varianti di answer relevance per ciascun record
    for i, rec in enumerate(records):
        q_emb = question_embs.get(i)
        g_emb = generated_embs.get(i)
        s_embs = synth_embs[i]

        # AR_ragas: media similarità con domande sintetiche
        if q_emb is not None and s_embs:
            sims = [float(q_emb @ syn) for syn in s_embs]
            rec["answer_relevance_ragas"] = round(sum(sims) / len(sims), 4)
        else:
            rec["answer_relevance_ragas"] = None

        # AR_direct: similarità diretta domanda ↔ risposta
        if q_emb is not None and g_emb is not None and rec.get("generated", "").strip():
            rec["answer_relevance_direct"] = round(float(q_emb @ g_emb), 4)
        else:
            rec["answer_relevance_direct"] = None

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Salvataggio
# ─────────────────────────────────────────────────────────────────────────────
def safe_avg(recs: list, key: str):
    """Media aritmetica dei valori non-None di *key* nei record, arrotondata a 4 decimali."""
    vals = [r[key] for r in recs if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def save_results(results: list, out_dir: Path):
    """Raggruppa i risultati per (model_id, dataset) e salva un JSON ciascuno."""
    # Raggruppamento dei record per coppia modello/dataset
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["model_id"], r["dataset"])].append(r)

    print(f"\n  Salvataggio {len(grouped)} file:")
    for (model_id, dataset), recs in sorted(grouped.items()):
        # Costruisce il percorso di output sostituendo '/' con '__' nello slug
        slug     = model_id.replace("/", "__")
        out_path = out_dir / slug / dataset / "ragass_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = len(recs)

        # Calcolo delle metriche aggregate per il summary
        summary = {
            "model_id":  model_id,
            "dataset":   dataset,
            "n_samples": n,
            # Answer Correctness
            "accuracy_ragas":          safe_avg(recs, "answer_correctness"),
            "accuracy_string":         round(sum(r["is_correct_string"] for r in recs) / n, 4),
            "ragas_string_agreement":  round(
                sum(r["answer_correctness"] == r["is_correct_string"] for r in recs) / n, 4
            ),
            # Faithfulness
            "faithfulness_avg":    safe_avg(recs, "faithfulness"),
            "avg_n_statements":    safe_avg(recs, "n_statements"),
            # Answer Relevance — due versioni
            "ar_ragas_avg":        safe_avg(recs, "answer_relevance_ragas"),
            "ar_direct_avg":       safe_avg(recs, "answer_relevance_direct"),
            # Context Relevance
            "context_relevance_avg": safe_avg(recs, "context_relevance"),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": recs},
                      f, ensure_ascii=False, indent=2)

        ac  = summary["accuracy_ragas"]
        fa  = summary["faithfulness_avg"]
        ar_r = summary["ar_ragas_avg"]
        ar_d = summary["ar_direct_avg"]
        cr  = summary["context_relevance_avg"]
        name = model_id.split("/")[-1]
        print(
            f"  {name:<30} | {dataset:<8} | "
            f"AC={ac:.3f}  F={fa:.3f}  "
            f"AR_r={ar_r if ar_r else 'N/A'}  "
            f"AR_d={ar_d:.3f}  CR={cr:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Punto di ingresso: carica i risultati intermedi, calcola embedding, salva risultati finali."""
    # Parsing degli argomenti da riga di comando
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true",
                        help="Forza CPU (default: GPU)")
    parser.add_argument("--input-file", type=str, default=None,
                        help=f"File JSON di input (default: {DEFAULT_INTERIM_PATH})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Cartella di output (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()

    # Determinazione dei percorsi di input/output
    interim_path = Path(args.input_file) if args.input_file else DEFAULT_INTERIM_PATH
    out_dir      = Path(args.output_dir) if args.output_dir else DEFAULT_OUT_DIR

    # Caricamento dei risultati intermedi della Fase 1
    print(f"Caricamento: {interim_path}")
    with open(interim_path, encoding="utf-8") as f:
        records = json.load(f)
    print(f"  Record: {len(records)}")

    with_qs = sum(1 for r in records if r.get("synthetic_questions"))
    print(f"  Con domande sintetiche: {with_qs}/{len(records)}")

    # Inizializzazione del modello di embedding
    device = "cpu" if args.cpu else "cuda"
    print(f"\nCaricamento embedder: {EMBED_MODEL} su {device.upper()}")
    embedder = SentenceTransformer(EMBED_MODEL, device=device)

    # Calcolo degli embedding e delle metriche AR
    print("\nCalcolo embedding...")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = compute_embeddings(records, embedder)

    # Salvataggio dei risultati raggruppati per modello/dataset
    save_results(results, out_dir)

    # Riepilogo globale delle metriche aggregate su tutti i record
    n   = len(results)
    ac  = [r["answer_correctness"]        for r in results if r.get("answer_correctness")        is not None]
    fa  = [r["faithfulness"]              for r in results if r.get("faithfulness")              is not None]
    ar_r = [r["answer_relevance_ragas"]   for r in results if r.get("answer_relevance_ragas")    is not None]
    ar_d = [r["answer_relevance_direct"]  for r in results if r.get("answer_relevance_direct")   is not None]
    cr  = [r["context_relevance"]         for r in results if r.get("context_relevance")         is not None]

    print(f"\n{'='*55}")
    print(f"RIEPILOGO GLOBALE — {n} record")
    if ac:   print(f"  Answer Correctness  : {sum(ac)/len(ac):.3f}")
    if fa:   print(f"  Faithfulness        : {sum(fa)/len(fa):.3f}")
    if ar_r: print(f"  AR_ragas (LLM+emb)  : {sum(ar_r)/len(ar_r):.3f}")
    if ar_d: print(f"  AR_direct (emb only): {sum(ar_d)/len(ar_d):.3f}")
    if cr:   print(f"  Context Relevance   : {sum(cr)/len(cr):.3f}  [dataset-level]")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
