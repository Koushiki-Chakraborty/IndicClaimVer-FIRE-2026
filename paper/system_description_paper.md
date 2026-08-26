# ClaimSense at IndicClaimVer@FIRE 2026: Retrieval-Augmented and Knowledge-Graph-Boosted Multilingual Claim Verification

**Authors:** Sayan De, Koushiki Chakraborty, Nirnit Roy, Anindita Dutta, Archismita Das
**Affiliation:** [Affiliation placeholder — fill in institution name(s)]
**Contact:** [Corresponding author email placeholder]
**Team:** ClaimSense

> This file is a plain-English draft of the full system description paper, written to be converted into the CEUR-WS LaTeX template afterward. No LaTeX markup is used here — headings, tables, and figure placeholders are written so they map cleanly onto `\section`, `\begin{table}`, and `\begin{figure}` blocks later. Two figure placeholders are left for Neo4j Browser screenshots (one per subtask) — marked clearly below — to be inserted once supplied.

---

## Abstract

We present the ClaimSense system submitted to the IndicClaimVer@FIRE 2026 shared task, which targets multilingual (English, Hindi, Bengali, and code-mixed) claim verification over the IndicClaimBuster dataset. We participated in both subtasks. For **Subtask 1** (closed-domain evidence retrieval and veracity prediction over a fixed evidence pool), our system combines a per-language BM25 baseline, dense multilingual retrieval using `intfloat/multilingual-e5-base`, and a Neo4j knowledge-graph-boosted reranking stage that reranks the dense top-10 candidates by inverse-document-frequency-weighted entity overlap, computed through a multi-stage Wikidata entity-linking cascade (exact match, code-mixed retry, Bengali morphological normalization, fuzzy search, semantic reranking, and an LLM fallback for the hardest cases). Veracity is predicted either by GPT-4o-mini reasoning over the top-1 retrieved evidence, or by a standalone MuRIL-based binary classifier fine-tuned with hyperparameters found via Optuna search. For **Subtask 2** (open-domain web-based verification), our system generates native-language and English search queries deterministically, retrieves evidence via the Tavily web search API, ranks it through a pipeline of BM25, multilingual sentence-embedding similarity, and a persistent Neo4j entity graph (optionally enriched with cross-lingual Wikidata QID linking), fuses the three rankings with Reciprocal Rank Fusion, fetches full article text for the top-3 fused results, and produces a grounded veracity verdict and justification using a large language model under a strict no-hallucination prompting regime. On the official test set, our submitted Subtask 1 system reaches a veracity macro-F1 of 78.05 (English), 85.88 (Hindi), 82.46 (Bengali), and 79.94 (code-mixed), and our Subtask 2 system reaches a veracity macro-F1 of 76.91 (English), 68.97 (Hindi), 62.92 (Bengali), and 36.51 (code-mixed). On our own development-set ablations, knowledge-graph-boosted retrieval is the strongest Subtask 1 retrieval configuration we tested (Success@3 = 0.8395, nDCG@3 = 0.5748 on the combined train+dev set, beating a dense-retrieval-only baseline at every entity-overlap weight tested in [0.01, 0.05]), and LLM-based verification is the strongest Subtask 2 configuration we tested (macro-F1 = 0.8039 on the 631-claim development set), outperforming a reused standalone MuRIL classifier (macro-F1 = 0.6820) on the same claims. We discuss language-specific weaknesses — Bengali and, most severely, code-mixed text are consistently the hardest buckets across every retrieval and verification method we tried, in both our own ablations and the official test-set results — along with our ranking-fusion design decisions and directions for future work.

---

## 1. Introduction

Automated fact verification for low-resource and code-mixed Indic languages faces two compounding difficulties: (i) claims and evidence are often expressed in a mix of scripts and registers — formal news prose, informal social-media phrasing, and code-mixed Hindi–Bengali–English sentences — and (ii) the evidence a verification system must reason over may either come from a small, fixed pool (a *closed* retrieval setting) or from the open web (an *open-domain* setting), each of which demands a different system design. The IndicClaimVer@FIRE 2026 shared task, built on the IndicClaimBuster dataset (Pal et al., 2025), evaluates both settings directly through two subtasks:

- **Subtask 1 (closed evidence retrieval + veracity prediction):** given a claim and a fixed pool of roughly 13,000 evidence passages, retrieve the top-3 most relevant passages and predict whether the claim is `SUPPORTS`ed or `REFUTES`d.
- **Subtask 2 (open-domain retrieval + veracity prediction):** given a claim only, retrieve supporting or refuting evidence from the live web, and produce a veracity verdict together with a natural-language justification grounded in that evidence.

Our team, ClaimSense, built independent but architecturally related pipelines for the two subtasks. Both pipelines share a design philosophy: combine a fast lexical signal (BM25), a semantic signal (multilingual sentence embeddings), and a symbolic, inspectable signal (a Neo4j knowledge graph built from extracted entities), rather than relying on a single neural reranker. We found this combination gives both stronger retrieval quality and a more debuggable system — every ranking decision can be traced back to an explicit score component (a BM25 score, a cosine similarity, or an entity-overlap count visible directly in the graph).

This paper documents both systems in full: data handling, retrieval architecture, knowledge-graph construction and entity linking, veracity prediction (both an LLM-based path and a standalone fine-tuned classifier), the ablations that shaped our final configuration, and the official and development-set results for both subtasks.

**Contributions.** (1) An entity-overlap knowledge-graph reranking stage for closed-domain retrieval that we show, through a controlled weight sweep, robustly improves over dense-retrieval-only ranking on both Success@3 and nDCG@3 across a wide range of its single hyperparameter. (2) A multi-stage, cost-aware entity-linking cascade for Indic-script text that combines free Wikidata lookups (exact match, code-mixed retry, Bengali morphological stripping, fuzzy search) with a last-resort, disk-cached LLM fallback used only when the free cascade is structurally weak or unconfident. (3) A deterministic (LLM-free) bilingual query-generation stage for open-domain retrieval that preserves exact numeric tokens (case counts, percentages, dates) shown empirically to matter for pinning retrieval to the correct day's article. (4) A Reciprocal-Rank-Fusion-based evidence ranker that combines lexical, semantic, and knowledge-graph signals without needing to calibrate their raw score scales against one another. (5) A grounded LLM verification prompt with explicit anti-hedging and anti-hallucination constraints, checked both by instruction and by post-hoc validation of the model's output. (6) A reuse analysis showing that the same fine-tuned MuRIL classifier, applied without modification to both subtasks, is consistently and substantially outperformed by LLM-based veracity prediction on both — a result we report honestly, including a methodological correction we made to our own development-set evaluation (described in Section 8.2) after discovering an earlier scoring inconsistency.

---

## 2. Related Work

Our system builds directly on the IndicClaimBuster resource (Pal, Jana, and Das, 2025), which introduced the multilingual claim–evidence–label dataset underlying both subtasks of this shared task; we describe our required citation of this work in Section 12.

Our retrieval design follows the now-standard "lexical + dense + fusion" pattern used across recent multilingual information retrieval systems: a probabilistic bag-of-words ranker (BM25, via the Okapi BM25 formulation) provides a strong, script-agnostic lexical baseline; a multilingual dense bi-encoder (`intfloat/multilingual-e5-base`) provides a semantic signal robust to vocabulary mismatch across languages; and, for Subtask 2, Reciprocal Rank Fusion (Cormack, Clarke, and Buettcher, 2009) combines multiple rank-based signals without requiring cross-signal score calibration. Our knowledge-graph reranking stage is inspired by entity-centric relevance signals used in knowledge-graph-augmented retrieval and fact-verification systems, but we deliberately keep the graph schema minimal (claim/evidence/entity nodes and mention/overlap edges) rather than modeling a full knowledge base, since our goal is a transparent ranking booster, not a general-purpose knowledge base. For entity linking, we build a cost-aware cascade over Wikidata's public SPARQL and `wbsearchentities` endpoints, escalating to a large language model only for the hardest, highest-value cases — a design choice made explicit and measured in Section 5.5.

For veracity prediction, we use two complementary approaches also common in the fact-verification literature: (i) prompting a general-purpose instruction-following LLM (GPT-4o-mini for Subtask 1, and for Subtask 2, a model from the Gemini family — see the backend note in Section 6.7) with a strict, evidence-grounded decision rule, and (ii) fine-tuning a smaller, dedicated multilingual encoder (MuRIL) as a lower-cost, offline-capable classifier. We compare both directly against each other on both subtasks.

---

## 3. Task Description

### 3.1 Subtask 1 — Closed Evidence Retrieval and Veracity Prediction

Given a claim `{ID, Text}` and a fixed pool of evidence passages `{ID, Evidence}` (approximately 13,333 items for the development/training split, 28,618 for the official test split), the system must return, for each claim, the three evidence passages ranked most-to-least relevant, plus a single veracity label, `SUPPORTS` or `REFUTES`. Systems are scored on retrieval quality — Success@3 and nDCG@3 against the single gold evidence item per claim — and on veracity accuracy/macro-F1 against the gold label. This is a *closed* setting: no live web access is permitted or needed; every claim's supporting or refuting evidence is guaranteed to exist somewhere in the fixed pool.

### 3.2 Subtask 2 — Open-Domain Retrieval and Veracity Prediction

Given a claim only (no fixed evidence pool), the system must retrieve its own supporting or refuting evidence from the open web, and produce a structured verdict record `{ID, Evidence, Prediction, Justification}`, where `Evidence` is a concise, evidence-grounded summary, `Prediction` is exactly `SUPPORTS` or `REFUTES`, and `Justification` explains the verdict with explicit reference to the retrieved evidence. Unlike Subtask 1, there is no fixed pool and no guarantee that any single passage directly confirms or denies the claim — the system must locate, rank, and reason over live web content itself.

### 3.3 Languages

Both subtasks cover four language conditions: English, Hindi, Bengali, and **code-mixed** text (informal claims that mix Latin and Indic scripts within a single sentence, common in social-media-sourced misinformation). Code-mixed and Bengali text are, across both of our subtasks and every retrieval method we tested, the consistently hardest buckets — a pattern we discuss in Section 9.

---

## 4. Datasets

### 4.1 Subtask 1

| Split | Claims | Evidence pool size | Notes |
|---|---:|---:|---|
| Train | 4,536 | 13,333 (shared) | `Label` distribution: 3,040 `SUPPORTS` / 1,496 `REFUTES` |
| Dev | 1,134 | 13,333 (shared) | Used for all ablations reported in Section 8.1 |
| Official test | 3,900 | 28,618 | 1,200 English + 1,200 Hindi + 1,200 Bengali + 300 code-mixed |

### 4.2 Subtask 2

| Split | Claims | Notes |
|---|---:|---|
| Dev | 631 | Gold `Label` provided; used for our reported development metrics |
| Official test | 1,009 | 299 English + 278 Hindi + 258 Bengali + 174 code-mixed |

For both subtasks, claim language is either provided directly by the source file (Subtask 2's four per-language test files, and our own filename-based tagging convention) or detected via a lightweight, character-count-weighted script classifier for Subtask 1 (Devanagari and Bengali Unicode ranges are weighted 2× relative to Latin characters, since a single Latin word inside an otherwise-Devanagari sentence should not flip the detected language). We additionally flag a claim as **code-mixed** whenever it contains both an Indic-script run and a Latin-script run of at least four characters.

---

## 5. System Architecture — Subtask 1

### 5.1 Pipeline Overview

```
                 ┌─────────────┐     ┌────────────────┐     ┌──────────────────────┐
   Claim ───────►│  Language   │────►│  BM25 baseline  │     │  Dense retrieval      │
                 │  detection  │     │ (per-language)  │     │  (multilingual-e5)    │
                 └─────────────┘     └────────────────┘     └──────────┬────────────┘
                                        (Gate-2 baseline,               │ top-10
                                         not in final chain)            ▼
                                                            ┌───────────────────────────┐
                                                            │  Knowledge-Graph-boosted   │
                                                            │  rerank (Neo4j)            │
                                                            │  entity overlap, IDF-      │
                                                            │  weighted, top-3 cut       │
                                                            └──────────┬─────────────────┘
                                                                       │ top-3 evidence
                                                                       ▼
                                                            ┌───────────────────────────┐
                                                            │  Veracity prediction        │
                                                            │  GPT-4o-mini (top-1 ev.)    │
                                                            │  OR MuRIL classifier        │
                                                            └──────────┬─────────────────┘
                                                                       ▼
                                                            {ID, top_3_evidence, Prediction}
```

The production pipeline is **Dense retrieval (top-10) → Knowledge-Graph-boosted rerank (top-3) → Veracity prediction**. BM25 is retained as a per-language lexical baseline used to validate that dense retrieval is actually the stronger signal before committing to it (see Section 8.1), but it is not part of the final scored chain.

### 5.2 Dense Retrieval

We encode every claim and every evidence passage with `intfloat/multilingual-e5-base` (768-dimensional embeddings), using the E5 family's documented asymmetric `"query: "` / `"passage: "` text prefixing. We selected E5 over an earlier multilingual sentence-embedding model (`paraphrase-multilingual-mpnet-base-v2`) specifically because that earlier model's own published training-language coverage does not include Bengali, which produced catastrophic Bengali dense-retrieval Success@3 (0.207) in an early validation gate; E5 is trained on mC4, which covers Bengali, and is purpose-built for asymmetric query/passage retrieval rather than paraphrase similarity. Embeddings are cached to disk (hashed by text content, namespaced by model name) so re-encoding is never repeated across pipeline runs. We evaluated two dense-retrieval variants — a full-pool cosine search and a language-partitioned search restricted to same-language evidence — and found them statistically indistinguishable (Section 8.1); we report the full-pool variant as our primary dense baseline.

### 5.3 Knowledge-Graph-Boosted Reranking

We build a Neo4j graph with `Claim`, `Evidence`, and `Entity` node types, connected by `MENTIONS` edges (from a claim or evidence passage to each entity it mentions) and `HAS_MENTION → LINKS_TO` edges recording the specific surface-form mention and its resolved Wikidata entity. Given a claim's dense top-10 candidates, we rerank them by a single Cypher query that computes each candidate's entity overlap with the claim, **weighted by inverse document frequency**: a shared entity that appears in only a few documents across the whole corpus contributes close to a full point of overlap score, while a shared entity that appears in thousands of documents (e.g., a country or a national political figure) contributes close to zero. This IDF weighting was itself an empirical fix: an earlier flat-count overlap scheme (every shared entity counted equally) *underperformed* plain dense retrieval (Success@3 0.7834 vs. 0.8312 on the full train+dev set), because common entities dominated the score without carrying real disambiguating value. After switching to IDF weighting, we swept the single scaling hyperparameter (`ENTITY_OVERLAP_WEIGHT`) over the range [0.0, 2.0] on the full 5,670-claim train+dev set and found a robust improving region: **every weight in [0.01, 0.05] beat dense-retrieval-only on both Success@3 and nDCG@3**, with a maximum at weight = 0.02 (Section 8.1, Table 3).

We additionally investigated a cluster-density penalty (DBSCAN-based disambiguation, penalizing candidates that fall inside a dense, low-specificity evidence cluster while sharing zero entity overlap with the claim). This was found to make retrieval **strictly worse at every nonzero penalty weight tested** — at production scale, DBSCAN with the tuned similarity threshold collapsed into a single giant cluster per language rather than isolating genuinely confusable sub-events, turning the penalty into an indiscriminate demotion rather than a targeted disambiguation signal. This component is implemented but disabled in the final system; we report the full experiment in Section 8.1 for completeness.

> **[FIGURE PLACEHOLDER — Subtask 1 Neo4j Knowledge Graph]**
> *A Neo4j Browser screenshot of the Subtask 1 graph is to be inserted here, showing `Claim`/`Evidence`/`Entity`/`Mention` nodes and their `MENTIONS`/`HAS_MENTION`/`LINKS_TO` relationships for a representative claim, illustrating the entity-overlap structure the reranking query traverses.*

### 5.4 Entity Extraction and Linking Cascade

Entities are extracted from claim and evidence text using `ai4bharat/IndicNER`, a token-classification model, with mentions re-sliced from the *original* text using the model's character offsets (expanded to full word boundaries) rather than the model's own reconstructed sub-word tokens — a fix for a confirmed corruption bug where sub-word reconstruction dropped Devanagari/Bengali diacritics or truncated proper nouns at sub-word boundaries (e.g., producing "Kam" instead of "Kamrup"). Each extracted mention is then resolved to a Wikidata QID through a five-stage cascade, run in order until a sufficiently strong candidate pool is found:

1. **Exact label/alt-label match** against Wikidata, in the mention's detected language.
2. **Code-mixed retry**: if the pool is still weak and the mention is Latin-script but the claim's language is not English, retry the exact match in English.
3. **Bengali morphological stripping**: a validated four-rule suffix-stripping table (e.g., stripping the emphatic/case markers *-ই*, *-র*, *-ে*, *-ের*) recovers entities whose surface form is inflected (e.g., "মহুয়ার" → "মহুয়া", which correctly resolves to the politician Mahua Moitra).
4. **Fuzzy search** (`wbsearchentities`), re-ranked to prefer candidates whose Wikidata label is a strict compound extension of the mention (surfacing, e.g., "Left Front" for the bare Bengali word "বাম", which otherwise ranks below generic dictionary senses of the same word).
5. **Type-aware filtering and semantic reranking**: candidates are filtered by whether Wikidata's own class hierarchy confirms the expected entity type (PER via direct instance-of-human; ORG/LOC via a transitive subclass check against organization/geographic-entity root classes), then re-scored by a weighted combination of Wikidata notability (sitelink count), a compound-match bonus, and the cosine similarity between the mention's local context sentence and the candidate's Wikidata description (both embedded with the same E5 model, min–max normalized within the candidate pool).

A final, **strictly last-resort** stage escalates unresolved or low-confidence mentions to GPT-4o-mini, which is asked only to produce a full English name or expansion for the mention (never a QID or description directly) — the result is then re-searched through the same Wikidata pipeline, so every final answer remains grounded in a genuine Wikidata lookup rather than an LLM guess. This stage is disabled by default (real API cost) and fires only when the free cascade is structurally weak (zero candidates of the expected entity type) or, when semantic reranking is active, when the top candidate does not clearly beat the runner-up. On a representative validation run, this fallback triggered for 22.9% of mentions, of which 77% produced a genuine correction. A second-tier variant of the same fallback additionally supplies the top-3 semantically related evidence passages as extra disambiguating context, used only for claim-side resolution.

Every network-facing stage of this cascade is disk-cached (candidate lists, type-membership flags, and LLM resolutions each in their own cache), with an explicit *cache-freshness* check that treats any cache entry predating a given bug fix as stale and re-fetches it — preventing a corrected pipeline from silently continuing to serve pre-fix data.

### 5.5 Veracity Prediction

We implement two independent veracity-prediction paths over the KG-boosted top-3 evidence:

- **GPT-4o-mini**, prompted with the claim and the top-1 retrieved evidence passage, asked to reply with exactly one word (`SUPPORTS` or `REFUTES`).
- **A standalone MuRIL classifier**: `google/muril-base-cased` fine-tuned as a binary sequence classifier, taking the claim and a single candidate evidence passage as a sentence pair (`max_length = 192`), trained with class-weighted cross-entropy to correct for the training set's SUPPORTS/REFUTES imbalance (weights 0.746 / 1.516, computed as `total / (2 × class_count)`). Hyperparameters were selected via a 15-trial Optuna search (learning rate, batch size, epoch count, weight decay, warmup ratio, and max gradient norm all searched jointly, optimizing dev macro-F1), yielding a final configuration of learning rate ≈ 1.80 × 10⁻⁵, 16 epochs, weight decay ≈ 0.0765, warmup ratio ≈ 0.0265, max gradient norm ≈ 1.20, with mixed-precision (`fp16`) training. The fine-tuned checkpoint is published on the Hugging Face Hub (`nirnit-13/muril-supports-refutes`) and is loaded directly for inference — there is no local training code path in the released pipeline; the classifier is trained once, offline, in a notebook, and reused thereafter purely as an inference-time model.

These two paths are **not merged**; we report both independently (Section 8.1) and, per the shared task's requirements, ensure every reported number corresponds to a single, specific, reproducible configuration.

---

## 6. System Architecture — Subtask 2

### 6.1 Pipeline Overview

```
Claim ──► Query Generation ──► Web Retrieval (Tavily) ──► Merge + Dedup ──► Evidence Filter
                                                                                   │
                                                                                   ▼
                                                                            BM25 Ranking
                                                                                   │
                                                                                   ▼
                                                                        Embedding Rerank (E5)
                                                                                   │
                                                                                   ▼
                                                        Entity Graph Ranking (Neo4j, + optional
                                                            Wikidata QID cross-lingual signal)
                                                                                   │
                                                                                   ▼
                                                            Reciprocal Rank Fusion (BM25 rank +
                                                                embedding rank + entity rank)
                                                                                   │
                                                                                   ▼
                                                        Full-page fetch (top-3 fused evidence)
                                                                                   │
                                                                                   ▼
                                                            LLM Verification (grounded prompt,
                                                             anti-hedging, retry-on-truncation)
                                                                                   │
                                                                                   ▼
                                                    {ID, Evidence, Prediction, Justification}
```

Unlike Subtask 1, this pipeline has no fixed evidence pool: every claim triggers a fresh web search, and every stage after retrieval operates over whatever the search API returns for that specific claim.

### 6.2 Query Generation

Because regional-language news sources are often indexed only in their own language, we generate **two** search queries per claim — one in the claim's original language, one in English — rather than relying on English-only search. Unlike an earlier version of this stage, query generation is now **fully deterministic** (no LLM call): entities are extracted with the same NER pipeline used for the knowledge-graph stage, stopwords are stripped using the `stopwordsiso` multilingual stopword lists (unioned across English, Hindi, and Bengali for code-mixed claims), and a short query is assembled from, in priority order, (1) exact numeric tokens found verbatim in the claim text (case counts, percentages, dates), (2) extracted entities kept as intact multi-word spans, and (3) remaining non-stopword keywords in order of appearance, capped at eight words. The numeric-token priority was added after diagnosing a recurring development-set failure pattern: claims stating a specific statistic were frequently retrieved against a *different* day's report on the same running story, because the exact figure was being crowded out of the word budget by generic, topically-recurring keywords. English translation (used both for the English query and for downstream embedding reranking) is produced via `deep-translator`'s free Google Translate wrapper, with a raw-text fallback on any translation failure.

### 6.3 Web Retrieval and Filtering

We retrieve results for both queries via the Tavily Search API, scoped to its `topic="news"` mode (falling back automatically to unscoped search on any error), merge and deduplicate the two result sets by URL, and apply a conservative evidence-filtering pass that removes clear non-article noise — cookie/terms-of-service/login pages (detected by URL pattern and exact-title match) and thin, near-content-free social-media links — while never removing an item that would leave the pool empty. This filtering stage was added after observing that such pages could still score competitively on keyword overlap alone and occupy a top-3 verification slot without contributing genuine evidentiary content.

### 6.4 BM25 and Embedding Reranking

The merged, filtered pool is scored twice: once with BM25 (Okapi variant, corpus-level IDF weighting, scored against the combined native + English query string, so a single ranking pass covers evidence in either language), and once with cosine similarity between the claim's English translation and each evidence item's title+snippet text, embedded with `intfloat/multilingual-e5-base` (falling back to `BAAI/bge-base-en-v1.5` if the preferred model cannot be loaded). Both stages append their raw score, min–max-normalized score, and rank to every evidence item, without modifying or removing any existing field — every pipeline stage is additive in this sense, which is what allows Reciprocal Rank Fusion (Section 6.6) to simply read off whichever rank fields it needs.

### 6.5 Entity Graph Ranking

As in Subtask 1, we build a Neo4j graph — here with `Claim {id, text, lang, created_at}`, `Evidence {url, title, snippet, retrieved_at}`, and `Entity {normalized_text, original_text, qid?}` nodes, connected by `MENTIONS` edges and a `MATCHES {overlap_raw, overlap_norm, entity_rank}` edge from each evidence item to the claim it was ranked against. Because the shared task's open-domain setting has no fixed evidence pool to pre-link, this graph is **surface-form-only by default** (entities are matched by a normalized string comparison, with a suffix-stripping and substring-containment fallback for Bengali/Hindi inflected forms), avoiding a per-claim SPARQL dependency on the retrieval-critical path. We layer an **optional** Wikidata QID cross-lingual linking signal on top: for each claim and evidence entity, we attempt a live Wikidata lookup (rate-limited, disk-cached, with a circuit breaker that disables further live lookups for the remainder of a run after a single HTTP 429) and compute a parallel QID-overlap count. An evidence item's effective overlap score is the maximum of its surface-form overlap and its QID overlap — this can only ever raise a candidate's score relative to surface-form-only behavior, never lower it, and gracefully degrades to pure surface-form matching if Wikidata is unreachable or rate-limited. This is what lets the system recognize, for example, that an English claim mentioning "Bamangachi" and a Bengali evidence passage mentioning "বামনগাছি" refer to the same real-world place, which no amount of string normalization alone can detect across scripts.

Unlike Subtask 1's graph, this graph is designed to **accumulate persistently** across an entire batch run (and across runs) — the same real-world entity or article is represented by exactly one shared node regardless of how many different claims retrieve or mention it, since nodes are keyed by stable values (normalized entity text, evidence URL) rather than anything claim-specific.

> **[FIGURE PLACEHOLDER — Subtask 2 Neo4j Knowledge Graph]**
> *A Neo4j Browser screenshot of the Subtask 2 graph is to be inserted here, showing the accumulated `Claim`/`Evidence`/`Entity` structure across multiple claims, illustrating a cross-claim shared-entity/shared-evidence pattern and (if enabled) a QID-linked cross-script entity pair.*

### 6.6 Reciprocal Rank Fusion

We combine the three per-item ranks — `bm25_rank`, `embedding_rank`, `entity_rank` — with the standard Reciprocal Rank Fusion formula,

RRF(*d*) = Σᵢ 1 / (*k* + *rᵢ*(*d*)),

with *k* = 60 (the conventional default from Cormack, Clarke, and Buettcher, 2009, also used as Elasticsearch's built-in RRF default). We fuse ranks rather than raw scores deliberately: BM25 scores are unbounded and corpus-dependent, embedding similarity is roughly bounded in [0, 1], and entity overlap is a small integer count — averaging or weighting these directly would let whichever signal happens to have the largest numeric range dominate the fusion, whereas ranks are scale-free by construction. Missing ranks (an item absent from one signal) are simply skipped rather than penalized with an artificial worst-case value.

### 6.7 LLM Verification

For the top-3 fusion-ranked evidence items, we fetch the full article text of each source URL (via `trafilatura`, with a BeautifulSoup boilerplate-stripping fallback, and a graceful fallback to the item's own search-result snippet on total fetch failure), and construct a single grounded verification prompt asking a large language model to produce a structured JSON record with three fields: a 100–150-word evidence summary, an exact `SUPPORTS`/`REFUTES` prediction, and a 100–120-word justification. The prompt enforces several strict constraints, checked both by instruction and by post-hoc validation of the model's actual output:

- **No outside knowledge.** The model must reason only over the supplied evidence text.
- **No hedging.** Because the task has exactly two labels and no "not enough evidence" option, the prompt explicitly forbids common hedging phrases ("insufficient evidence," "cannot verify," etc.) in either output field; if one slips through anyway, the response is discarded and regenerated (up to two additional retries).
- **Language-matched output.** The `Evidence` and `Justification` fields must be written in the claim's own language (English, Hindi, Bengali, or a code-mixed register matching the claim's own script-mixing pattern), while the `Prediction` field is always required to remain the exact English string `SUPPORTS` or `REFUTES`, regardless of the other two fields' language — this was itself a fix for an observed failure where a Bengali claim originally produced fully English output, because the model was never explicitly told to match the input language.
- **Truncation handling.** Responses cut off by the token budget (more likely for Bengali/Hindi text, which tokenizes less efficiently than English) are detected explicitly via the model's finish-reason signal and retried from scratch, rather than patched — we deliberately never attempt to "complete" a truncated sentence, since that is itself a form of the hallucination this stage is designed to avoid.

**Backend note.** Our verification stage currently calls a Gemini-family model (accessed through Google's `google-genai` SDK) as its LLM backend. During development, this stage was temporarily switched to OpenAI's GPT-4o-mini to work around a Gemini free-tier daily quota limitation while processing the full 1,009-claim test set, and was subsequently switched back to Gemini once a suitable, sufficiently-available Gemini model tier was identified. We report this transparently because it affects which specific model produced which historical result file; the results reported in Section 8.2 were produced with the pipeline's current, Gemini-based configuration.

### 6.8 Standalone MuRIL Classifier (Subtask 2)

As a lower-cost, offline-capable alternative to LLM-based verification, we additionally evaluate the **identical** fine-tuned MuRIL classifier described in Section 5.5, reused as-is for Subtask 2 without any retraining on Subtask 2 data. For each claim, we classify the claim text against the title+snippet text of its single top-1 fusion-ranked evidence item (the same evidence representation already used by the BM25/embedding/entity-graph stages, chosen so this classification path never needs to re-fetch full article text for every candidate). This is a genuinely separate prediction path from the LLM-based `run_pipeline.py` submission — the two are never merged into a single default output, and we report both independently in Section 8.2.

### 6.9 Submission Aggregation

`build_submission.py` aggregates every claim's finished verification record into one final JSON array, validating that every prediction label is exactly `SUPPORTS` or `REFUTES` before writing. The pipeline is fully resumable: every stage checks for its own already-written output file and skips claims already completed, so an interrupted 1,009-claim batch run over a rate-limited external API can simply be restarted rather than redone from scratch, and a single claim's failure at any stage is logged and skipped rather than aborting the batch.

---

## 7. Experimental Setup

| Component | Subtask 1 | Subtask 2 |
|---|---|---|
| Dense/embedding model | `intfloat/multilingual-e5-base` | `intfloat/multilingual-e5-base` (fallback: `BAAI/bge-base-en-v1.5`) |
| NER model | `ai4bharat/IndicNER` | `Davlan/xlm-roberta-base-ner-hrl` |
| Graph database | Neo4j 5.26 (Community, APOC plugin) | Neo4j 5.26 (Community, APOC plugin) |
| Lexical ranker | `rank_bm25` (Okapi BM25) | `rank_bm25` (Okapi BM25) |
| Web search API | — (closed pool) | Tavily Search API (`topic="news"`) |
| Veracity LLM | GPT-4o-mini | Gemini family (see Section 6.7) |
| Standalone classifier | MuRIL (`google/muril-base-cased`, fine-tuned) | Same checkpoint, reused |
| Classifier max sequence length | 192 tokens | 192 tokens |
| Classifier training precision | `fp16` | `fp16` (same checkpoint) |
| Hyperparameter search | Optuna, 15 trials | Optuna, 15 trials (same search) |
| Entity-overlap rerank weight | 0.02 (swept in [0.0, 2.0]) | N/A (surface-form + optional QID overlap, unweighted max-combine) |
| Evidence pool size | 13,333 (dev/train) / 28,618 (test) | Open web (Tavily, `TOP_K_PER_QUERY = 6` per query) |
| RRF constant *k* | N/A | 60 |
| Fusion signals | N/A | BM25 rank, embedding rank, entity rank |

All dense/embedding-model encodings are cached to disk to avoid redundant recomputation across ablations and repeated pipeline runs. Both pipelines run entity extraction and Neo4j graph writes as resumable batch stages — network-facing lookups (Wikidata SPARQL, `wbsearchentities`, Wikipedia redirect resolution) are all disk-cached with an explicit staleness check tied to specific documented bug fixes, so a corrected pipeline is guaranteed not to silently keep serving pre-fix cached results.

---

## 8. Results

### 8.1 Subtask 1

**Table 1 — Official test-set results for Subtask 1 (ClaimSense, Submission 1, as reported by the shared task organizers' results sheet).** The sheet reports Subtask 1 in two parts: **Subtask 1a — Evidence Retrieval** (Success@3, MRR@3, nDCG@3) and **Subtask 1b — Veracity Prediction** (Precision, Recall, Macro-F1), both reproduced verbatim below, per language.

| Language | Success@3 (1a) | MRR@3 (1a) | nDCG@3 (1a) | Precision (1b) | Recall (1b) | Macro-F1 (1b) |
|---|---:|---:|---:|---:|---:|---:|
| English | 79.58 | 49.67 | 57.54 | 78.26 | 78.08 | 78.05 |
| Hindi | 88.67 | 51.99 | 61.28 | 88.26 | 86.08 | 85.88 |
| Bengali | 67.00 | 38.99 | 46.08 | 84.42 | 83.50 | 82.46 |
| Code-mixed | 26.67 | 22.33 | 23.44 | 80.35 | 80.00 | 79.94 |

**Table 2 — Development-set retrieval ablation (Success@3 / nDCG@3), our own internal validation, run to select the final production configuration.**

| Method | Scope (n) | Hindi | Bengali | English | Code-mixed | Overall |
|---|---:|---|---|---|---|---|
| BM25 | 1,134 (dev) | 0.889 / 0.607 | 0.670 / 0.488 | 0.796 / 0.629 | 0.662 / 0.470 | 0.768 / 0.563 |
| Dense (full-pool) | 1,134 (dev) | 0.875 / 0.584 | 0.682 / 0.488 | 0.869 / 0.623 | 0.773 / 0.512 | 0.818 / 0.566 |
| Dense (language-partitioned) | 1,134 (dev) | 0.875 / 0.575 | 0.687 / 0.490 | 0.874 / 0.626 | 0.773 / 0.512 | 0.821 / 0.565 |
| Dense-only | 5,670 (train+dev) | 0.882 / 0.567 | 0.739 / 0.539 | 0.882 / 0.634 | 0.767 / 0.505 | 0.831 / 0.571 |
| **KG-boosted (production)** | 5,670 (train+dev) | 0.883 / 0.568 | 0.756 / 0.547 | 0.889 / 0.635 | 0.780 / 0.512 | **0.8395 / 0.5748** |

**Table 3 — Entity-overlap weight sweep** (KG-boosted reranking, all 5,670 train+dev claims, one Neo4j pass reused across every weight tried).

| Weight | Success@3 | nDCG@3 | |
|---:|---:|---:|---|
| 0.00 | 0.8312 | 0.5713 | (= dense-only exactly; sanity check) |
| 0.01 | 0.8376 | 0.5746 | beats dense-only |
| 0.015 | 0.8383 | 0.5749 | beats dense-only |
| **0.02** | **0.8395** | **0.5748** | **best — beats dense-only on both** |
| 0.025 | 0.8377 | 0.5742 | beats dense-only |
| 0.03 | 0.8379 | 0.5736 | beats dense-only |
| 0.04 | 0.8388 | 0.5738 | beats dense-only |
| 0.05 | 0.8381 | 0.5733 | beats dense-only |
| 0.10 | 0.8231 | 0.5664 | below dense-only again |
| 0.20 | 0.8035 | 0.5542 | far below (flat-count-scale weight, too strong once IDF-weighted) |

**Table 4 — Development-set veracity prediction**, both scored on the same KG-boosted top-3 evidence.

| Model | Scope (n) | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| MuRIL classifier | 1,134 (dev only) | 0.8660 | 0.8480 |
| GPT-4o-mini | 1,134 (dev only) | **0.9198** | **0.9075** |
| GPT-4o-mini | 5,670 (train+dev) | — | 0.9236 |

**Discussion.** KG-boosted reranking beats dense-retrieval-only on both metrics at every entity-overlap weight tried in [0.01, 0.05] — a real, robust improving region, not a single lucky configuration. GPT-4o-mini outperforms our fine-tuned MuRIL classifier on veracity prediction by a wide margin (macro-F1 0.9075 vs. 0.8480 on the same 1,134-claim dev set and same evidence). Bengali and code-mixed are consistently the weakest retrieval buckets across every method tested, a known weak point of general-purpose multilingual embedding models and, for code-mixed text specifically, of script-based language detection generally.

### 8.2 Subtask 2

**Table 5 — Official test-set results for Subtask 2 (ClaimSense, Submission 1, as reported by the shared task organizers' results sheet).** These are the officially evaluated Precision/Recall/Macro-F1 scores for our submitted system (the Gemini-based `run_pipeline.py` LLM verification path, Section 6.7 — this is the system we actually submitted; the standalone MuRIL classifier of Section 6.8 was not submitted and has no official row of its own).

| Language | Precision | Recall | Macro-F1 |
|---|---:|---:|---:|
| English | 77.02 | 76.93 | 76.91 |
| Hindi | 68.86 | 68.76 | 68.97 |
| Bengali | 62.88 | 63.09 | 62.92 |
| Code-mixed | 54.06 | 57.00 | 36.51 |
| **Unweighted average across languages** | **65.71** | **66.45** | **61.33** |

The unweighted per-language average is our own computation from the four rows above (the results sheet does not publish a separate pooled "Overall" figure for this system); we report it only as a convenient single-number summary, not as an official metric in its own right. The most striking result here is **code-mixed Macro-F1 collapsing to 36.51** — far below its own Precision/Recall (54.06/57.00) and far below every other language's Macro-F1 — indicating a highly imbalanced or degenerate prediction pattern specifically on code-mixed claims at official test time (e.g., the system defaulting toward one label far more often than the true label distribution warrants), a more severe version of the code-mixed weakness we also observe throughout Subtask 1's retrieval results (Section 8.1) and discuss further in Section 9.

**Table 6 — Development-set veracity prediction (631 claims, gold labels), our own internal comparison between the submitted LLM path and the non-submitted, reused MuRIL classifier** — one row per claim, comparing each system's actual final `Prediction` against gold `Label`. This is a controlled ablation on our own held-out dev split, distinct from Table 5's official per-language test-set numbers.

| System | Accuracy | Macro-F1 | F1 (SUPPORTS) | F1 (REFUTES) |
|---|---:|---:|---:|---:|
| **LLM verification (Gemini) — the submitted system** | **0.8082** | **0.8039** | **0.8331** | **0.7747** |
| MuRIL classifier (reused from Subtask 1, not submitted) | 0.7005 | 0.6820 | 0.7586 | 0.6054 |

**A methodological note we report transparently.** An earlier internal evaluation of the MuRIL classifier's development-set performance reported a substantially lower accuracy (0.5563), because it scored every individual (claim, evidence) pair the classifier was run against (1,891 rows: up to three candidate evidence items per claim), rather than the single final prediction actually used per claim, as the LLM path's own metric does. We identified and corrected this inconsistency before finalizing the comparison above; the corrected number (0.7005 accuracy) is the one we report as authoritative, computed on exactly the same 631-claim, one-row-per-claim basis as the LLM path. We note this correction explicitly in the interest of reporting only genuinely comparable numbers, per the shared task's own requirement that all reported results correspond to a consistent, reproducible evaluation.

We additionally note one remaining asymmetry between the two systems compared in Table 6: the LLM verification path reads the *full fetched article text* of its top-3 evidence sources, while the MuRIL classifier — to avoid re-fetching every candidate URL a second time — is scored against only the top-ranked item's search-result title and snippet. Part of the accuracy gap in Table 6 may therefore reflect this input-quality difference rather than model capability alone; we flag this as a controlled follow-up experiment for future work (Section 10). Note also that Table 5 (official test set) and Table 6 (our own dev split) are not directly comparable to each other — different claims, and Table 5 is broken down per language while Table 6 is pooled — so the gap between, e.g., Table 6's 0.8039 overall dev macro-F1 and Table 5's 61.33 average official test macro-F1 should not be read as a single clean number describing "how much worse the system does on the official test set"; the code-mixed collapse in Table 5 is doing most of that work on its own, as the per-language breakdown shows.

---

## 9. Error Analysis and Discussion

**Code-mixed veracity prediction collapses on the official Subtask 2 test set.** Table 5's code-mixed row shows Precision 54.06 and Recall 57.00 — already the weakest of the four languages, but not dramatically so — alongside a Macro-F1 of only 36.51, a much larger drop than Precision/Recall alone would suggest. A Macro-F1 this far below both its own Precision/Recall and every other language's Macro-F1 is a signature of a degenerate or heavily skewed prediction pattern (for instance, the system defaulting toward one label far more often than the true code-mixed label distribution warrants, which depresses the *minority* class's F1 sharply even while Precision/Recall averaged another way look only moderately weak). We had no comparably sized, per-language-labeled internal held-out set for Subtask 2's official test distribution specifically to catch this before submission — our own development-set numbers (Table 6) are pooled across languages, not broken out by language, which is precisely the kind of blind spot that let this pattern go undetected internally. We flag targeted, per-language dev-set monitoring — not just a single pooled dev metric — as a concrete process fix for future submissions (Section 10).

**Bengali and code-mixed retrieval are the hardest conditions across both subtasks.** In Subtask 1, code-mixed Success@3 lags every other language bucket by a wide margin regardless of retrieval method (e.g., 0.780 KG-boosted vs. 0.889 for English); Bengali is consistently second-weakest. We attribute this to two compounding factors: general-purpose multilingual sentence embeddings are typically trained on far less Bengali data than English or Hindi, and code-mixed claims by definition straddle two or more scripts within a single sentence, which both weakens single-language keyword overlap for BM25 and can confuse script-based language routing.

**Entity-overlap reranking is a real but narrow improvement.** The KG-boosted reranking stage's gain over dense-retrieval-only is real and validated by a full weight sweep (Table 3), but it is a modest absolute improvement (Success@3 +0.0083 at the best weight) — appropriate for a signal designed to *disambiguate near-ties* in the dense top-10, not to override dense retrieval's primary ranking. We view this as the correct scope for a symbolic signal layered on top of a strong neural baseline, rather than a limitation of the approach.

**Cluster-density penalties do not help at production scale**, despite being a reasonable hypothesis (candidates in the same "confusable event cluster" as several other retrieved items might be systematically over- or under-weighted). Our controlled experiment (Section 5.3) showed this degrades monotonically as the penalty weight increases, because the clustering threshold that worked on a small, single-story validation set collapsed into one dominant cluster per language at full corpus scale. We view this as a useful negative result: a disambiguation signal that only works at a smaller scale than the deployed system is not safe to deploy without re-validating the clustering threshold at the actual production scale.

**LLM-based veracity prediction consistently and substantially outperforms our fine-tuned classifier**, on both subtasks (Subtask 1: macro-F1 0.9075 vs. 0.8480; Subtask 2: macro-F1 0.8039 vs. 0.6820). We consider two likely, non-exclusive explanations: (i) GPT-4o-mini and Gemini are general-purpose models with substantially larger pretraining exposure to multilingual and code-mixed text than a 180M-parameter fine-tuned MuRIL classifier, and (ii) the LLM paths reason over richer context (a full evidence passage or full fetched article, versus the classifier's shorter, `max_length=192`-token-truncated input) — a difference we explicitly flag in Section 8.2 as a confound worth isolating in future work rather than treating this comparison as a clean measure of "LLM vs. fine-tuned model" capability alone.

**The verification stage's strict anti-hedging design is a deliberate response to the task's binary-only label space.** Because FIRE 2026 provides no "insufficient evidence" label, a system that defaults to a mechanical "evidence is thin → REFUTES" rule would systematically misclassify genuinely under-evidenced SUPPORTS claims. Our prompt explicitly bans this shortcut and checks for it post-hoc; we consider this a necessary design choice given the task's label space, not an optional refinement.

---

## 10. Limitations and Future Work

- **Wikidata linking coverage.** Our entity-linking cascade is grounded entirely in Wikidata, which has known coverage gaps for minor regional entities, informal organization names, and very recent events — a mention with no Wikidata presence at all cannot be resolved by any stage of our cascade, LLM fallback included (the fallback only re-searches Wikidata with a better name; it does not invent an entity that genuinely is not there).
- **Evidence representation parity for the Subtask 2 classifier comparison.** As noted in Section 8.2, the MuRIL classifier and the LLM verification path are not currently evaluated on identical evidence text. A controlled follow-up giving the classifier the same full-article text as the LLM path (at the cost of re-fetching every candidate URL) would isolate model capability from input-quality as the source of their performance gap.
- **Event-clustering disambiguation** remains an open problem at production scale; our experiment shows the *current* DBSCAN-based approach does not help, not that no clustering-based signal could help with a better-calibrated threshold or a different clustering algorithm.
- **Live web retrieval instability (Subtask 2).** Web search results for a given query can change over time (articles removed, paywalled, or superseded), meaning Subtask 2 evaluation is not perfectly reproducible run-to-run in the way Subtask 1's fixed-pool evaluation is — a structural property of the open-domain setting rather than a defect in our system, but worth stating explicitly.
- **Code-mixed-specific modeling.** Both subtasks currently treat code-mixed text largely as "apply every relevant language's tooling and union the results" (stopwords, entity linking) rather than using any code-mixing-aware model or normalization step; given that code-mixed claims are the weakest bucket across every method we tested, a dedicated code-mixed normalization or embedding fine-tuning pass is a promising direction.
- **Per-language development monitoring, not just a single pooled dev metric.** The official Subtask 2 code-mixed Macro-F1 collapse (Section 9) was not visible in our own pooled, all-languages development metric before submission. Tracking Precision/Recall/Macro-F1 separately per language during development — mirroring exactly how the shared task itself reports results — would very likely have surfaced this specific failure mode in time to address it before the official run, rather than after.

---

## 11. Conclusion

We presented ClaimSense's system for both subtasks of IndicClaimVer@FIRE 2026. Our Subtask 1 system combines dense multilingual retrieval with an IDF-weighted, Wikidata-grounded knowledge-graph reranking stage, validated by a full hyperparameter sweep to robustly improve over dense retrieval alone, paired with either an LLM-based or a fine-tuned-classifier veracity predictor; on the official test set it reaches a veracity Macro-F1 of 78.05–85.88 across English, Hindi, and Bengali, and 79.94 on code-mixed. Our Subtask 2 system generates bilingual search queries deterministically, retrieves and ranks open-web evidence through a three-signal Reciprocal-Rank-Fusion pipeline (lexical, semantic, and knowledge-graph-based), and produces grounded, anti-hallucination-constrained veracity verdicts via an LLM, with a reused standalone classifier evaluated internally as a lower-cost alternative (not submitted); on the official test set it reaches a veracity Macro-F1 of 62.92–76.91 across English, Hindi, and Bengali, but collapses to 36.51 on code-mixed — the single most actionable finding of this paper, and one we did not detect internally before submission due to using only a pooled, not per-language, development metric. Across both subtasks, LLM-based veracity prediction meaningfully outperforms our fine-tuned MuRIL classifier on our own internal comparison, and Bengali/code-mixed text remains the consistent weak point for both retrieval quality and, on the official Subtask 2 test set specifically, veracity prediction itself — together the clearest, best-evidenced directions for future system improvement.

---

## Acknowledgments

We thank the IndicClaimVer@FIRE 2026 organizers for creating and running this shared task, and for providing the IndicClaimBuster dataset underlying both subtasks.

---

## 12. References

Pal, P., Jana, S. K., and Das, D. (2025). IndicClaimBuster: A Multilingual Claim Verification Dataset. In *Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL 2025)*, pp. 2478–2489, Mumbai, India. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics. https://aclanthology.org/2025.ijcnlp-long.133/ , DOI: 10.18653/v1/2025.ijcnlp-long.133

*(This citation is required by the shared task organizers and must be included in the final paper — see BibTeX entry below.)*

```bibtex
@inproceedings{pal-etal-2025-indicclaimbuster,
    title = "{I}ndic{C}laim{B}uster: A Multilingual Claim Verification Dataset",
    author = "Pal, Pritam and
      Jana, Shyamal Krishna and
      Das, Dipankar",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    pages = "2478--2489",
    url = "https://aclanthology.org/2025.ijcnlp-long.133/",
    doi = "10.18653/v1/2025.ijcnlp-long.133"
}
```

**Other references to add in the final LaTeX version** (technical components used but not yet formally cited above — add proper citations for each before submission):

- Robertson, S. and Zaragoza, H. — the Okapi BM25 probabilistic ranking function.
- Cormack, G. V., Clarke, C. L. A., and Buettcher, S. (2009) — Reciprocal Rank Fusion, for the RRF formula used in Section 6.6.
- Wang, L. et al. — the `multilingual-e5-base` embedding model (Text Embeddings by Weakly-Supervised Contrastive Pre-training / the E5 family).
- Khanuja, S. et al. — MuRIL: Multilingual Representations for Indian Languages, the base model fine-tuned for our veracity classifiers.
- The `ai4bharat/IndicNER` model card/paper (AI4Bharat), for the Subtask 1 NER model.
- The `Davlan/xlm-roberta-base-ner-hrl` model card, for the Subtask 2 NER model.
- OpenAI — GPT-4o-mini model documentation, for Subtask 1 veracity prediction and Subtask 2's development-period backend note.
- Google DeepMind — the Gemini model family documentation, for Subtask 2's LLM verification backend.
- Vrandečić, D. and Krötzsch, M. — Wikidata: a free collaborative knowledgebase, for our entity-linking target.
- Neo4j, Inc. — Neo4j graph database documentation, for our knowledge-graph implementation.
- Barbaresi, A. — `trafilatura`: a web scraping library for text discovery and extraction.
- Ester, M. et al. — DBSCAN, the density-based clustering algorithm referenced in Section 5.3's disabled event-clustering experiment.

---

## Author checklist before LaTeX conversion

- [ ] Confirm final author order and add institutional affiliations + emails.
- [ ] Insert the Subtask 1 Neo4j graph screenshot (Section 5.3 placeholder).
- [ ] Insert the Subtask 2 Neo4j graph screenshot (Section 6.5 placeholder).
- [x] Official Subtask 1 (Table 1) and Subtask 2 (Table 5) results inserted, from the organizers' results spreadsheet, ClaimSense / Submission 1.
- [ ] Double-check the official Table 1 and Table 5 numbers against the final, locked leaderboard before submission (these were pulled from a results spreadsheet linked from the organizers' results page at draft time; re-verify no later revision changed them, and confirm whether "Submission 1" is in fact the submission you want reported, if more than one was made).
- [ ] Add full bibliographic details for the "other references" list above.
- [ ] Convert this document into the CEUR-WS Overleaf template, targeting 6–10 pages including references.
- [ ] Circulate the copyright form to every listed author once the organizers share it.
