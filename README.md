# IndicClaimVer @ FIRE 2026: Team ClaimSense

Multilingual claim verification for English, Hindi, Bengali and code-mixed text, built for the IndicClaimVer shared task at FIRE 2026 (IndicClaimBuster dataset). Given a claim, the system retrieves evidence and predicts whether the evidence `SUPPORTS` or `REFUTES` it.

| Subtask | Setting | Evidence source | Retrieval | Verdict |
| ------- | ------- | --------------- | --------- | ------- |
| 1 | Closed domain | Fixed evidence pool | multilingual-e5 dense retrieval, reranked with Wikidata entity overlap from a Neo4j KG | GPT-4o-mini or fine-tuned MuRIL |
| 2 | Open domain | Live web (Tavily) | BM25, embedding and KG entity rankings merged with Reciprocal Rank Fusion | Gemini or fine-tuned MuRIL |

## Contents

* [Architecture](#architecture)
* [Knowledge graph](#knowledge-graph)
* [Results](#results)
* [Repository layout](#repository-layout)
* [Setup](#setup)
* [Running the pipelines](#running-the-pipelines)
* [Veracity classifier](#veracity-classifier)
* [Submission format](#submission-format)
* [Team](#team)

## Architecture

**Subtask 1: closed domain**

![Subtask 1 architecture](images/subtask1_architecture.png)

**Subtask 2: open domain**

![Subtask 2 architecture](images/subtask2_architecture.png)

## Knowledge graph

Both pipelines store entities in Neo4j. Subtask 1 links `Claim` and `Evidence` nodes to Wikidata `Entity` nodes through `MENTIONS` edges. Subtask 2 uses the same `Claim`, `Evidence`, `Entity` and `MENTIONS` structure, with web search results as `Evidence` and entities keyed by normalized text instead of a Wikidata QID.

**Subtask 1 graph**

![Subtask 1 Neo4j graph](images/neo4j_subtask1.png)

**Subtask 2 graph**

![Subtask 2 Neo4j graph](images/neo4j_subtask2.png)

## Results

**Official test set**

| Subtask | Metric | English | Hindi | Bengali | Code-mixed |
| ------- | ------ | ------: | ----: | ------: | ---------: |
| 1 | Success@3 (retrieval) | 79.58 | 88.67 | 67.00 | 26.67 |
| 1 | Macro-F1 (veracity) | 78.05 | 85.88 | 82.46 | 79.94 |
| 2 | Macro-F1 (veracity) | 76.91 | 68.97 | 62.92 | 36.51 |

**Dev set, veracity Macro-F1**

| Subtask | Claims | MuRIL | LLM |
| ------- | -----: | ----: | --: |
| 1 | 1,134 | 0.848 | 0.908 (GPT-4o-mini) |
| 2 | 631 | 0.682 | 0.804 (Gemini) |

**Subtask 1 retrieval (train + dev, 5,670 claims)**

| Retriever | Success@3 | nDCG@3 |
| --------- | --------: | -----: |
| Dense (multilingual-e5-base) | 0.831 | 0.571 |
| Dense + KG entity rerank (final) | **0.840** | **0.575** |

More retrieval baselines (BM25, BGE-M3, IndicSBERT) and per-language scores are in [`subtask1_pipeline/README.md`](subtask1_pipeline/README.md).

## Repository layout

```
.
├── README.md
├── requirements.txt          shared Python dependencies
├── docker-compose.yml        shared Neo4j 5 instance (bolt://localhost:7687)
├── images/                   architecture and KG diagrams
│
├── subtask1_pipeline/        Subtask 1: closed domain retrieval + verification
│   ├── README.md             how to run, scripts, results
│   ├── config.py
│   ├── data/                 train / dev claims and evidence pool
│   └── subtask1/             package: retrievers, entity linking, KG builder
│       ├── scripts/          dev set entry points (numbered setup scripts, kg_pipeline.py)
│       ├── officialTestData/ test data and test set entry points
│       ├── experiments/      KG weight sweep, cluster penalty experiment
│       ├── notebook/         Colab notebooks for BGE-M3 and IndicSBERT retrieval
│       └── results/          devResult/, testResult/, score/
│
├── subtask2_pipeline/        Subtask 2: open domain web retrieval + verification
│   ├── README.md             how to run, stages, outputs
│   ├── run_pipeline.py       single entry point
│   ├── compute_f1.py         Macro-F1 scorer
│   ├── data/                 dev and test claims
│   ├── src/                  pipeline stages
│   └── output/               per-claim outputs and submissions
│
└── classifier/               MuRIL veracity classifier shared by both subtasks
    ├── README.md
    ├── notebooks/            hyperparameter search, training, prediction
    ├── model_selection/      MuRIL vs IndicBERTv2 vs mBERT vs XLM-R
    └── tokenizer_config/     tokenizer and config files of the released checkpoint
```

## Setup

Requirements: Python, Docker (for Neo4j), and API keys for the services you plan to use.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
docker-compose up -d              # Neo4j browser at http://localhost:7474
```

In each pipeline folder, copy `.env.example` to `.env` and fill in the keys.

| Key | Used by | Needed for |
| --- | ------- | ---------- |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | both | KG stages (defaults match `docker-compose.yml`) |
| `HF_TOKEN` | Subtask 1 | gated Hugging Face models |
| `OPENAI_API_KEY` | both | GPT-4o-mini verdicts (Subtask 1) and justifications (Subtask 2) |
| `TAVILY_API_KEY` | Subtask 2 | web search |
| `GEMINI_API_KEY` | Subtask 2 | Gemini verifier only |

The two pipelines use different graph schemas: Subtask 1 keys `Entity` nodes by Wikidata QID, Subtask 2 by normalized text. To keep the graphs apart, set a different `NEO4J_DATABASE` in each pipeline's `.env`.

## Running the pipelines

Short version below. Each pipeline README covers every stage, flag and output file.

**Subtask 1** (dev set). Details in [`subtask1_pipeline/README.md`](subtask1_pipeline/README.md).

```bash
cd subtask1_pipeline
python subtask1/scripts/03_build_kg_st1.py
python subtask1/scripts/kg_pipeline.py dense
python subtask1/scripts/kg_pipeline.py finalize --with-gpt
```

**Subtask 2** (test set by default). Details in [`subtask2_pipeline/README.md`](subtask2_pipeline/README.md).

```bash
cd subtask2_pipeline
python run_pipeline.py                         # MuRIL verifier (default)
python run_pipeline.py --verifier gemini       # Gemini verifier, as reported in the paper
python compute_f1.py --gold data/dev_subtask2.json --pred output/GPT_output/dev_submission.json
```

## Veracity classifier

A MuRIL model (`google/muril-base-cased`) fine-tuned on Subtask 1 claim and evidence pairs, with class-weighted loss and Optuna hyperparameter search. The same checkpoint is used in both subtasks without retraining.

* Checkpoint: [nirnit-13/muril-supports-refutes](https://huggingface.co/nirnit-13/muril-supports-refutes)
* Label mapping: `{0: SUPPORTS, 1: REFUTES}`. This is the reverse of `config.LABEL2ID` in the pipelines, so always read labels from the checkpoint's `id2label`.
* Model selection, dev Macro-F1: MuRIL 0.813, IndicBERTv2 0.801, mBERT 0.730, XLM-R 0.692.

Training notebooks and model selection results are in [`classifier/`](classifier/README.md).

## Submission format

One file per language, named `<Language>_subtask{1,2}.json` with `Language` in `English`, `Hindi`, `Bengali`, `CodeMix`.

| Subtask | Fields |
| ------- | ------ |
| 1 | `ID`, `top_3_evidence`, `Prediction` |
| 2 | `ID`, `Evidence`, `Prediction`, `Justification` |

`Prediction` must be exactly `SUPPORTS` or `REFUTES`.

## Team

Koushiki Chakraborty, Archismita Das, Anindita Dutta, Nirnit Roy, Sayan De, Imon Mukherjee

Code maintained by [Tarka Lab](https://github.com/TARKA-Lab).
