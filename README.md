
# Beyond Language Specialization: A RAG Evaluation of Italian LLMs

Code, evaluation scripts, and aggregated results for the CLiC-it 2026 paper
**"Beyond Language Specialization: A RAG Evaluation of Italian LLMs"**
(Agabiti, Trappolini, Rago, Romagnoli, Giannone, Silvestri).

We benchmark eleven models — Italian-specialized and general-purpose open-weight —
across five datasets in two languages (TriviaQA, Natural Questions, BioASQ, UniQA,
PharmaQA.IT) in a fixed-retriever RAG pipeline, scored with a reimplementation of the
RAGAS metrics under **two independent LLM judges** (GPT-4o-mini as primary, Qwen3-8B
as secondary).

> Main finding: Italian-specialized models remain competitive on English benchmarks
> but do not outperform general-purpose alternatives on Italian corpora. Velvet-14B,
> the strongest Italian-specialized model, ranks fifth across all settings.

---

## What this repository contains

This repo releases the **pipeline** (generation → sample preparation → judging →
aggregation) and the **aggregated results** that reproduce the tables of the paper.
It does not vendor the large intermediate artifacts (raw generations, per-record judge
outputs), which are regenerable from the code, nor the source datasets, which are
linked to their original distributions.

```
.
├── src/                      LLM wrappers (backends)
│   ├── llm.py                    local HuggingFace models
│   ├── llm_vllm_chat.py          local models served via vLLM
│   ├── llm_velvet.py             Velvet family via Almawave API (ENG + PharmaQA)
│   └── llm_velvet_uniqa.py       Velvet via API, UniQA system prompt (Italian)
│
├── generate_rag_eng.py                    generation — English (local)
├── generate_rag_pharmaqa.py               generation — PharmaQA (local)
├── generate_rag_uniqa_vllm.py             generation — UniQA (local + Velvet dispatch)
├── generate_rag_velvet_eng_unified.py     generation — English (Velvet)
├── generate_rag_velvet_pharmaqa.py        generation — PharmaQA (Velvet)
│
├── prepare_ragass_data_eng.py             build frozen eval samples — English
├── prepare_ragass_data_pharmaqa.py        build frozen eval samples — PharmaQA
├── prepare_ragass_data_uniqa.py           build frozen eval samples — UniQA
│
├── run_ragass_phase1_eng.py               Qwen3-8B judge, phase 1 (LLM calls) — English
├── run_ragass_phase1_pharmaqa.py          Qwen3-8B judge, phase 1 — PharmaQA
├── run_ragass_phase1_uniqa_fixed.py       Qwen3-8B judge, phase 1 — UniQA
├── run_ragass_phase2.py                   Qwen3-8B judge, phase 2 (AR embeddings)
│
├── llm_as_a_judge_gpt4o_eng.py            GPT-4o-mini judge, Answer Correctness — English
├── llm_as_a_judge_gpt4o_uniqa.py          GPT-4o-mini judge, AC — UniQA
├── llm_as_a_judge_gpt4o_pharmaqa.py       GPT-4o-mini judge, AC — PharmaQA
├── llm_as_a_judge_gpt4o_rag_metrics_eng.py       GPT-4o-mini, F/AR/CR — English
├── llm_as_a_judge_gpt4o_rag_metrics_uniqa.py     GPT-4o-mini, F/AR/CR — UniQA
├── llm_as_a_judge_gpt4o_rag_metrics_pharmaqa.py  GPT-4o-mini, F/AR/CR — PharmaQA
│
├── collect_summaries.py      distil Qwen3-8B results  -> results/qwen_judge/*.json
├── collect_gpt.py            distil GPT-4o-mini results -> results/gpt4o_judge/*.json
├── make_tables.py            reprint the paper tables from results/
│
├── results/                  aggregated per-table summaries (committed, ~20 kB total)
│   ├── qwen_judge/{english,uniqa,pharmaqa}.json     Tables 5, 6, 7
│   └── gpt4o_judge/{english,uniqa,pharmaqa}.json    Tables 1, 2, 3
│
└── data/                     inputs (see "Data" below — not committed)
```

---

## Which script produces which table

The paper reports results under two judges. **GPT-4o-mini is the primary judge**
(Tables 1–3); **Qwen3-8B is the secondary judge** used for validation (Tables 5–7).

| Table | Dataset(s)                 | Judge        | Scripts |
|-------|----------------------------|--------------|---------|
| 1     | TriviaQA, NQ, BioASQ       | GPT-4o-mini  | `llm_as_a_judge_gpt4o_eng.py` + `..._rag_metrics_eng.py` |
| 2     | UniQA                      | GPT-4o-mini  | `llm_as_a_judge_gpt4o_uniqa.py` + `..._rag_metrics_uniqa.py` |
| 3     | PharmaQA.IT                | GPT-4o-mini  | `llm_as_a_judge_gpt4o_pharmaqa.py` + `..._rag_metrics_pharmaqa.py` |
| 5     | TriviaQA, NQ, BioASQ       | Qwen3-8B     | `run_ragass_phase1_eng.py` + `run_ragass_phase2.py` |
| 6     | UniQA                      | Qwen3-8B     | `run_ragass_phase1_uniqa_fixed.py` + `run_ragass_phase2.py` |
| 7     | PharmaQA.IT                | Qwen3-8B     | `run_ragass_phase1_pharmaqa.py` + `run_ragass_phase2.py` |

The aggregated numbers behind every table are in `results/`. To reprint them:

```bash
python3 make_tables.py
```

---

## Reproducibility: what runs locally vs. what needs an API

| Component | Requires | Reproducible by a third party? |
|-----------|----------|-------------------------------|
| Local model generation (gemma, granite, qwen, llama, ministral, fastweb, villanova, minerva) | GPU + vLLM/transformers | **Yes** |
| Qwen3-8B judge (Tables 5–7) | GPU + vLLM | **Yes** |
| Velvet generation | Almawave API key | No — proprietary API |
| GPT-4o-mini judge (Tables 1–3) | OpenAI-compatible API key | No — external API |

Because Velvet and GPT-4o-mini are behind APIs, their outputs cannot be regenerated
without the corresponding credentials. This is why the **aggregated results are
committed**: they let anyone verify the reported numbers directly, even for the
API-only components.

All local models are served with greedy decoding (`temperature=0.0`) and
`repetition_penalty=1.1` where supported. Models with a dedicated chat template
(Gemma, Granite, Llama, Ministral, FastwebMIIA, Villanova) use `/v1/chat/completions`;
the others use raw completion. Qwen thinking mode is suppressed via `/no_think`.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or: conda create -n ragass python=3.11
pip install -r requirements.txt
cp .env.example .env      # then fill LLM_API_URL / LLM_API_TOKEN for Velvet + GPT-4o-mini
```

`.env` is only needed for the API components (Velvet generation, GPT-4o-mini judge).
The local pipeline (local models + Qwen3-8B judge) does not read it.

### vLLM servers

Two components use vLLM, with different settings. Serve one model at a time on a
single RTX 3090 (24 GB); stop the server (Ctrl+C) before running phase 2, which needs
the GPU for `bge-m3` embeddings.

**Generation on UniQA** (long answers, up to ~25k-token context):

```bash
vllm serve <model_id> --port 8000 --dtype bfloat16 \
    --gpu-memory-utilization 0.92 --max-model-len 32768 \
    --max-num-seqs 8 --disable-log-requests
```

**Qwen3-8B judge** (phase 1, secondary judge):

```bash
vllm serve Qwen/Qwen3-8B --port 8000 --dtype bfloat16 \
    --gpu-memory-utilization 0.90 --max-model-len 8192 \
    --max-num-seqs 64
```

---

## Data

The five datasets are **not redistributed here**; download them from their original
sources and place them under `data/dataset/`:

- **TriviaQA, NQ, BioASQ** — pre-retrieved passages from Trappolini et al. (paper ref. [9]).
- **UniQA** — Siragusa & Pirrone (ref. [8]).
- **PharmaQA.IT** — Zeinalipour et al. (ref. [4]).

Expected layout (paths hard-coded in the scripts):

```
data/dataset/
├── triviaqa_dataset.json
├── nq_dataset.json
├── bioasq_dataset.json
├── uniqa_it_dataset.json
└── pharmaqa_dataset.json
```

**Frozen evaluation samples and full aggregated results** (our own artifacts, produced
with `seed=42`) are attached to the GitHub Release **v1.0** as `.zip` archives.

---

## Data pipeline and formats

The repository operates on **already-retrieved** data. Retrieval from the raw corpora
(Wikipedia/KILT, PubMed, UniQA and PharmaQA documents) with `bge-m3` + FAISS is
upstream and **not included in this release**: the pipeline starts from the
pre-retrieved dataset files. Retrieval code will be added in a later update.

```
data/dataset/<name>.json               input: question + gold + ranked passages
        │
        │  generate_rag_*.py           prompt each model with the top-k passages
        ▼
data/gen_res/<model>/<dataset>/top5/*.pkl
        │
        │  prepare_ragass_data_*.py    freeze a common sample set (seed 42),
        │                              inline the retrieved context
        ▼
data/ragass/ragass_samples_<group>.json   flat: one record per model × example
        │
        │  run_ragass_phase1_* / llm_as_a_judge_gpt4o_*   judge scores each record
        ▼
interim / per-metric JSON
        │
        │  run_ragass_phase2 (Qwen)  or  collect_gpt.py (GPT)
        ▼
results/<judge>/<group>.json           aggregated per model × dataset → paper tables
```

### Format 1 — dataset input (`data/dataset/<name>.json`)

A JSON **list** of records. This is the starting point the scripts consume:

```json
{
  "example_id": "sfq_1589",
  "question": "...",
  "answers": ["gold 1", "gold 2"],
  "passages": [
    {"rank": 0, "is_relevant": true, "text": "..."}
  ]
}
```

- `answers` is always a **list**, even for a single gold reference.
- `passages` is pre-retrieved and ranked ascending by `rank`; ENG/UniQA carry 25
  passages, PharmaQA carries 5. The scripts take `passages[:k]` — they do **not**
  retrieve, so the list must already be the ranked top-k.
- Required per record: `example_id`, `question`, `answers`, `passages`.
  Required per passage: `text`, `is_relevant`.

### Format 2 — frozen evaluation samples (`ragass_samples_<group>.json`)

The output of `prepare_ragass_data_*`, and the input the judges consume. A flat JSON
list with **one record per (model, example)**, with the retrieved context inlined:

```json
{
  "example_id": "sfq_1589",
  "dataset": "bioasq",
  "model_id": "ibm-granite/granite-4.1-8b",
  "question": "...",
  "answers": ["..."],
  "generated": "the model's answer",
  "context": "[Document 1] ...\n\n[Document 2] ...",
  "is_correct_string": true,
  "n_relevant": 2
}
```

The English samples file holds all three English datasets together (tagged by the
`dataset` field); the phase-1 script filters per dataset with `--dataset`.

---

## Running the pipeline

The pipeline has three stages. `k=5` throughout (Minerva-7B uses `k=3` on PharmaQA
and is excluded from UniQA, due to its 4,096-token context window).

### 1. Generation

Produces per-model answers under `data/gen_res/<model>/<dataset>/top5/`.

```bash
# English (local models)
python generate_rag_eng.py --llm_id ibm-granite/granite-4.1-8b --datasets triviaqa nq bioasq --k_values 5

# PharmaQA (local models)
python generate_rag_pharmaqa.py --llm_id ibm-granite/granite-4.1-8b

# UniQA — requires a vLLM server; dispatches Velvet to the API wrapper automatically
python generate_rag_uniqa_vllm.py --llm_id google/gemma-4-E2B-it

# Velvet (API) — English and PharmaQA use dedicated scripts; UniQA reuses the script above
python generate_rag_velvet_eng_unified.py --llm_id velvet-14b --datasets triviaqa nq bioasq --k_values 5
python generate_rag_velvet_pharmaqa.py --llm_id velvet-14b
python generate_rag_uniqa_vllm.py --llm_id velvet-14b       # Velvet on UniQA
```

### 2. Prepare frozen evaluation samples

Intersects the example IDs available across all models and samples a fixed set
(seed 42), attaching the retrieved context. For English, a balanced 334/333/333
subsample of 1,000 QA pairs.

```bash
python prepare_ragass_data_eng.py      --output-file data/ragass/ragass_samples_eng.json
python prepare_ragass_data_pharmaqa.py --output-file data/ragass/ragass_samples_pharmaqa.json
python prepare_ragass_data_uniqa.py    --output-file data/ragass/ragass_samples_uniqa_it.json
```

### 3. Judge and aggregate

**Secondary judge — Qwen3-8B** (local, via vLLM). Phase 1 makes the LLM calls;
phase 2 computes Answer Relevance embeddings with `BAAI/bge-m3`.

```bash
# start: vllm serve Qwen/Qwen3-8B --port 8000 --dtype bfloat16 --max-model-len 8192

# phase 1 — English (context cap differs by dataset: 8000 for TriviaQA/NQ, 12000 for BioASQ)
python run_ragass_phase1_eng.py --input-file data/ragass/ragass_samples_eng.json --dataset triviaqa --ctx-max-chars 8000  --output-dir data/ragass/qwen_eng
python run_ragass_phase1_eng.py --input-file data/ragass/ragass_samples_eng.json --dataset nq       --ctx-max-chars 8000  --output-dir data/ragass/qwen_eng
python run_ragass_phase1_eng.py --input-file data/ragass/ragass_samples_eng.json --dataset bioasq   --ctx-max-chars 12000 --output-dir data/ragass/qwen_eng

# phase 1 — Italian
python run_ragass_phase1_pharmaqa.py    --input-file data/ragass/ragass_samples_pharmaqa.json --output-dir data/ragass/qwen_pharmaqa
python run_ragass_phase1_uniqa_fixed.py --input-file data/ragass/ragass_samples_uniqa_it.json --output-dir data/ragass/qwen_uniqa

# phase 2 — stop vLLM first, then run per interim file
python run_ragass_phase2.py --input-file <interim_from_phase1>.json --output-dir data/ragass/results
```

**Primary judge — GPT-4o-mini** (OpenAI-compatible API). Answer Correctness and the
F/AR/CR metrics are computed by separate scripts per dataset.

```bash
python llm_as_a_judge_gpt4o_eng.py            --input-file data/ragass/ragass_samples_eng.json      --output-dir data/llm_as_a_judge/eng
python llm_as_a_judge_gpt4o_rag_metrics_eng.py --input-file data/ragass/ragass_samples_eng.json     --output-dir data/llm_as_a_judge/eng_rag_metrics
# ... likewise for uniqa and pharmaqa
```

### Distil results into the committed summaries

```bash
python3 collect_summaries.py --source data/ragass --out results/qwen_judge
python3 collect_gpt.py --out results/gpt4o_judge \
  --eng    "data/llm_as_a_judge/eng/answer_correctness_accuracy_by_model_dataset.json:data/llm_as_a_judge/eng_rag_metrics/rag_metrics_summary_by_model_dataset.json" \
  --uniqa  "data/llm_as_a_judge/uniqa/answer_correctness_accuracy_by_model_dataset.json:data/llm_as_a_judge/uniqa_rag_metrics/rag_metrics_summary_by_model_dataset.json" \
  --pharma "data/llm_as_a_judge/pharmaqa/answer_correctness_accuracy_by_model_dataset.json:data/llm_as_a_judge/pharmaqa_rag_metrics/rag_metrics_summary_by_model_dataset.json"
```

---

## Notes and caveats

- **Two generation backends.** Local models run via `transformers` (English, PharmaQA)
  or via vLLM (UniQA, where long answers make batched serving worthwhile). Both use the
  same prompting and decoding; the split is an efficiency choice, not a methodological one.
- **Answer Relevance.** The reported AR is the RAGAS variant: mean cosine similarity
  (`bge-m3`) between the question and five synthetic questions generated by the judge.
- **Context Relevance** is a property of the fixed retriever, roughly constant per corpus;
  it is a system-level diagnostic, not a model-discriminating metric.
- **velvet-25b** and other exploratory runs are not part of the paper; only Velvet-2B
  and Velvet-14B are evaluated.
- Generation and evaluation prompts are reproduced verbatim in Appendix A of the paper.

---

## Citation

```bibtex
@inproceedings{agabiti2026beyond,
  title     = {Beyond Language Specialization: A RAG Evaluation of Italian LLMs},
  author    = {Agabiti, Riccardo and Trappolini, Giovanni and Rago, Salvatore and
               Romagnoli, Raniero and Giannone, Cristina and Silvestri, Fabrizio},
  booktitle = {Proceedings of the Twelfth Italian Conference on Computational
               Linguistics (CLiC-it 2026)},
  year      = {2026},
  address   = {Palermo, Italy}
}
```

Repository: <https://github.com/ragabiti/beyond-language-specialization>

