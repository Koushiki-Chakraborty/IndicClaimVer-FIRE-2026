# IndicClaimVer @ FIRE 2026 — Team ClaimSense

Multilingual claim verification (English, Hindi, Bengali, code-mixed) on the IndicClaimBuster dataset. Both subtasks combine BM25, multilingual embeddings and a Neo4j knowledge graph, with a fine-tuned MuRIL veracity classifier.

- **Subtask 1** (closed-domain): retrieve evidence from a fixed pool, rerank with the KG, predict veracity.
- **Subtask 2** (open-domain): search the live web, fuse ranking signals, predict veracity.

## Architecture

**Subtask 1**

![Subtask 1 Architecture](images/subtask1_architecture.png)

**Subtask 2**

![Subtask 2 Architecture](images/subtask2_architecture.png)

## Knowledge Graph (Neo4j)

**Subtask 1**

![Subtask 1 Neo4j Graph](images/neo4j_subtask1.png)

**Subtask 2**

![Subtask 2 Neo4j Graph](images/neo4j_subtask2.png)

## Structure

```
subtask1_pipeline/   # Subtask 1 code, data, results
subtask2_pipeline/   # Subtask 2 code, data, output
images/              # diagrams
requirements.txt     # shared dependencies
docker-compose.yml   # shared Neo4j
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # source venv/bin/activate on Linux/macOS
pip install -r requirements.txt
docker-compose up -d             # Neo4j at bolt://localhost:7687
```

In each pipeline folder, copy `.env.example` to `.env` and fill in the keys.

## Run

**Subtask 1**

```bash
cd subtask1_pipeline
python subtask1/scripts/03_build_kg_st1.py
python subtask1/scripts/kg_pipeline.py dense
python subtask1/scripts/kg_pipeline.py finalize --with-gpt
```

**Subtask 2**

```bash
cd subtask2_pipeline
python run_pipeline.py
python compute_f1.py --gold data/dev_subtask2.json --pred output/final_submission.json
```

See each pipeline's own README for details.

## Results (official test, Macro-F1)

| Subtask | English | Hindi | Bengali | Code-mixed |
|---|---|---|---|---|
| 1 | 78.05 | 85.88 | 82.46 | 79.94 |
| 2 | 76.91 | 68.97 | 62.92 | 36.51 |

## Team

Sayan De, Koushiki Chakraborty, Archismita Das, Nirnit Roy, Anindita Dutta
