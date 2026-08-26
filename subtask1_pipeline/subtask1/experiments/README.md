# Experiments

Exploratory work, kept separate from the actual pipeline
(`subtask1/scripts/kg_pipeline.py`, `subtask1/retriever.py`) on purpose —
nothing in here is imported by or affects the real submission
(`subtask1/results/dense_gpt/`) or the main KG results
(`subtask1/results/kg/`).

## Entity overlap weighted by inverse document frequency (`kg_weighted_overlap_reranker.py` / `kg_weighted_overlap_pipeline.py`)

**Motivation.** The real KG-boost (`KGBoostedRetriever.score_precomputed()`)
weights every shared entity equally — sharing "Delhi" or "Modi" (appears
in thousands of documents) counts the same as sharing a rare, specific
entity. Measured on the full production graph, this flat-count approach
underperformed plain Dense retrieval:

| Approach | Success@3 | nDCG@3 |
|---|---|---|
| Dense-only | 0.8312 | 0.5713 |
| KG-boost, flat overlap (`results/kg/`) | 0.7834 | 0.5417 |

**Fix tried:** weight each shared entity by inverse document frequency
instead of a flat count — an entity that appears in few documents
contributes close to 1, one that appears in many contributes close to
0. Read-only against the same already-linked graph, no new linking, no
writes.

**Weight sweep, all 5,670 train+dev claims, one Neo4j pass reused across
every weight tried.** Runnable and re-checked, not just a typed-in table
— `kg_weight_sweep.py` reuses `_entity_document_frequency_overlap()`
from `kg_weighted_overlap_reranker.py` (the exact function the validated
weight=0.02 reranker uses) and writes `weight_sweep_results.txt`,
checked into this folder, so a reviewer can rerun it and get the same
numbers below:

| Weight | Success@3 | nDCG@3 | |
|---|---|---|---|
| 0.0 | 0.8312 | 0.5713 | (= Dense-only exactly — sanity check the code is right) |
| 0.01 | 0.8376 | 0.5746 | beats Dense-only |
| 0.015 | 0.8383 | 0.5749 | beats Dense-only |
| **0.02** | **0.8395** | **0.5748** | **best — beats Dense-only on both** |
| 0.025 | 0.8377 | 0.5742 | beats Dense-only |
| 0.03 | 0.8379 | 0.5736 | beats Dense-only |
| 0.04 | 0.8388 | 0.5738 | beats Dense-only |
| 0.05 | 0.8381 | 0.5733 | beats Dense-only |
| 0.1 | 0.8231 | 0.5664 | below Dense-only again |
| 0.2 | 0.8035 | 0.5542 | far below (this is where the flat-count boost sits) |

**Result: a real win, and a robust one — every weight in [0.01, 0.05]
beat Dense-only on both metrics, not just one lucky point.** Best found:
`weight=0.02` — Success@3 +0.0083, nDCG@3 +0.0035 over Dense-only.
Deliberately kept as its own constant (`OVERLAP_WEIGHT` in
`kg_weighted_overlap_reranker.py`), separate from `config.ENTITY_OVERLAP_WEIGHT`
(0.2 — tuned for flat-count overlap, far too strong once overlap is
IDF-weighted instead).

**What this does NOT include:** the cluster-penalty half of the real
KG-boost. That was tried separately (see the main README's KG section)
and didn't hold up at production scale either — clustering degenerated
into one giant cluster per language regardless of threshold tried, so
this experiment only tests the entity-overlap half, deliberately.

**Not yet done:** wiring this into the actual GPT-classification step
or the official submission — this experiment is retrieval-only
(`Success@3`/`nDCG@3`), matching the current scope of the exploration.
Results in `subtask1/results/kg_test/` (`results.json`, `summary.txt`).

**Note:** this experiment predates the production integration of the
same idea into `KGBoostedRetriever._entity_overlap_counts()`
(`subtask1/retriever.py`) with `config.ENTITY_OVERLAP_WEIGHT = 0.02` —
kept here as the historical record of how the weight was found.

**Run it:**
```bash
python subtask1/experiments/kg_weight_sweep.py               # reproduces the table above -> weight_sweep_results.txt
python subtask1/experiments/kg_weighted_overlap_pipeline.py
python subtask1/experiments/kg_weighted_overlap_pipeline.py --limit 50   # quick test first
```

## Cluster-density penalty on top of the production scoring (`kg_cluster_penalty_experiment.py`)

**Question:** now that entity overlap is IDF-weighted and beats Dense-only
on its own (above), does stacking the cluster-density penalty back on
top of it help any further? Fully read-only — no `EventCluster` nodes
are written anywhere; clustering is computed in memory from Evidence
embeddings fetched read-only, using the exact same
`cluster_language_subset()` DBSCAN call `event_clustering.py` uses, just
without the write step. The production graph, and anything else
currently running against it, is untouched.

**Penalty formula tested** — the exact one that used to be in
`score_precomputed()` before it was removed (recovered from git
history): a candidate loses `penalty_weight` if it sits in a dense
cluster *and* shares zero weighted entity overlap with the claim.

| Penalty weight | Success@3 | nDCG@3 | |
|---|---|---|---|
| 0.0 | 0.8395 | 0.5748 | (= current production exactly — sanity check) |
| 0.02 | 0.8310 | 0.5694 | worse |
| 0.05 | 0.8168 | 0.5606 | worse |
| 0.1 | 0.8093 | 0.5561 | worse |
| 0.15 | 0.8093 | 0.5561 | worse (plateaus — penalty saturates) |
| 0.2 | 0.8093 | 0.5561 | worse |
| 0.3 | 0.8093 | 0.5561 | worse |

**Result: no, it doesn't help — every nonzero weight tried made both
metrics strictly worse, monotonically.** Root cause, measured directly
in this same run: at the current DBSCAN threshold (`CLUSTER_SIMILARITY_THRESHOLD=0.85`),
94-95% of evidence per language still lands inside *some* dense cluster
(4245/4527 English, 4224/4361 Bengali, 4211/4445 Hindi) — the same
"one giant cluster per language" degeneration already documented in the
main README's Known Limitations, confirmed again here. With clustering
that broad, the penalty fires on most zero-overlap candidates
indiscriminately rather than isolating genuinely confusable sub-events,
so it demotes plenty of otherwise-fine evidence along with the
actually-confusable cases. Confirms the decision already made in
`retriever.py` (cluster-penalty disabled) was correct, now specifically
re-checked against the current, improved entity-overlap scoring rather
than the old flat-count version.

**Run it:**
```bash
python subtask1/experiments/kg_cluster_penalty_experiment.py   # -> cluster_penalty_sweep_results.txt
```
