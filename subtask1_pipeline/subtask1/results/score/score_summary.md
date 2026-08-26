# Subtask 1 — Score Summary

All scores below come from the KG-boosted retrieval pipeline (`../devResult/kg*`). Files in this folder:

## `kg_retrieval_summary.txt`
KG-boosted retrieval quality, overall. Copy of `../devResult/kg/summary.txt`.

| n (train+dev) | Success@3 | nDCG@3 |
|---|---|---|
| 5670 | 0.8395 | 0.5748 |

## `kg_retrieval_by_language.txt`
Same KG-boosted retrieval, broken down by language. Computed from `../devResult/kg/result.json`.

| Language | Success@3 | nDCG@3 | n |
|---|---|---|---|
| Hindi | 0.8827 | 0.5682 | 1432 |
| Bengali | 0.7556 | 0.5465 | 978 |
| English | 0.8887 | 0.6349 | 1967 |
| Code-mixed | 0.7804 | 0.5122 | 1293 |
| **Overall** | **0.8395** | **0.5748** | **5670** |

## `kg_classifier_evaluation_metrics.json`
Veracity accuracy of the standalone MuRIL classifier on the KG-boosted top-3 evidence, dev set only. Copy of `../devResult/kg_classifier/evaluation_metrics.json`.

| n | Accuracy | Macro F1 | F1 SUPPORTS | F1 REFUTES |
|---|---|---|---|---|
| 1134 | 0.8660 | 0.8480 | 0.9003 | 0.7957 |

## `kg_gpt_evaluation_metrics.json` / `kg_gpt__summary.txt`
Veracity accuracy of GPT-4o-mini on the same KG-boosted top-3 evidence, filtered to the same 1134 dev claims as `kg_classifier` (the full train+dev GPT score stays in `../devResult/kg_gpt/`, not duplicated here). Computed from `../devResult/kg_gpt/kg_gpt_detailed.json`.

| n | Accuracy | Macro F1 | F1 SUPPORTS | F1 REFUTES |
|---|---|---|---|---|
| 1134 | 0.9198 | 0.9075 | 0.9412 | 0.8738 |

---

## Comparison (both on the same 1134 dev claims, same evidence)

| Model | Accuracy | Macro F1 |
|---|---|---|
| MuRIL classifier | 0.8660 | 0.8480 |
| GPT-4o-mini | **0.9198** | **0.9075** |

GPT-4o-mini wins on both metrics, mainly from stronger REFUTES recall (0.8015 vs 0.7532).
