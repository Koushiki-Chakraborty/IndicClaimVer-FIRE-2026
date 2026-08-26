"""
STEP 6: Embedding-based Reranking (SEMANTIC).

Single responsibility: given the claim's English translation and the
list of already-BM25-ranked evidence items (bm25.py's output), embed
the claim and each evidence item (title + snippet), score every item
by cosine similarity to the claim, normalize those scores, rank by
them, and return the SAME evidence items with three new fields added:

    embedding_score_raw, embedding_score_norm, embedding_rank

All existing fields from bm25.py's output (title, url, snippet,
bm25_score_raw, bm25_score_norm, bm25_rank) are preserved unchanged.

No entity overlap, no credibility weighting, no fusion, no Knowledge
Graph -- this module does exactly one thing: semantic similarity
reranking. Nothing is hardcoded to a specific claim, query, or file
path -- the core function takes plain data in and returns plain data
out; the optional CLI runner takes a claim folder path as an argument.

Model: BAAI/bge-m3 (preferred -- multilingual, handles English, Hindi,
Bengali, and code-mixed evidence text natively) with automatic fallback
to BAAI/bge-base-en-v1.5 if bge-m3 can't be loaded (no internet, model
unavailable, out of memory, etc.), so the pipeline degrades instead of
crashing.
"""
import json
from pathlib import Path

import numpy as np

_model = None
_model_name_used = None

_PREFERRED_MODEL = "intfloat/multilingual-e5-base"
_FALLBACK_MODEL = "BAAI/bge-base-en-v1.5"


def _get_model():
    """Lazily loads the embedding model once per process. Tries the
    preferred multilingual model first; falls back to an English-only
    model if that fails to load, so a missing/unavailable model doesn't
    crash the whole pipeline."""
    global _model, _model_name_used
    if _model is not None:
        return _model

    from sentence_transformers import SentenceTransformer

    try:
        _model = SentenceTransformer(_PREFERRED_MODEL)
        _model_name_used = _PREFERRED_MODEL
        print(f"[embedding_rerank] Loaded embedding model: {_PREFERRED_MODEL}")
    except Exception as e:
        print(f"[embedding_rerank] WARNING: could not load {_PREFERRED_MODEL} ({e}). "
              f"Falling back to {_FALLBACK_MODEL}.")
        _model = SentenceTransformer(_FALLBACK_MODEL)
        _model_name_used = _FALLBACK_MODEL
        print(f"[embedding_rerank] Loaded embedding model: {_FALLBACK_MODEL}")

    return _model


def _embed(texts: list) -> np.ndarray:
    """Encodes a list of strings into L2-normalized embedding vectors.
    Normalizing at encode time means cosine similarity reduces to a
    plain dot product later -- simpler, faster, no extra dependency."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def _minmax_normalize(scores: list) -> list:
    """Scales raw similarity scores to 0-1. If all scores are equal
    (including all-zero), returns all 0.5 to avoid divide-by-zero /
    false signal -- same convention used in bm25.py."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def rerank_with_embeddings(claim_text_english: str, ranked_evidence: list) -> list:
    """Pure, dynamic reranking function -- no file I/O, no hardcoded
    claim/query/path. Caller supplies the data.

    claim_text_english: the claim's English translation (already
    produced upstream by query_generation.py -- this module does no
    translation of its own).

    ranked_evidence: the "ranked_evidence" list from bm25.py's output
    -- each item already has title/url/snippet/bm25_* fields.

    Returns a NEW list, sorted by embedding similarity descending, with
    embedding_score_raw / embedding_score_norm / embedding_rank added to
    every item. All existing fields are preserved unchanged.
    """
    if not ranked_evidence:
        return []

    evidence_texts = [
        f"{item.get('title', '')} {item.get('snippet', '')}"
        for item in ranked_evidence
    ]

    claim_embedding = _embed([claim_text_english])[0]
    evidence_embeddings = _embed(evidence_texts)

    # Both sides are L2-normalized, so dot product == cosine similarity.
    raw_scores = [float(np.dot(claim_embedding, ev)) for ev in evidence_embeddings]
    norm_scores = _minmax_normalize(raw_scores)

    combined = sorted(
        zip(ranked_evidence, raw_scores, norm_scores),
        key=lambda item: -item[1],
    )

    reranked = []
    for rank, (item, raw, norm) in enumerate(combined, start=1):
        new_item = dict(item)  # copy -- preserves every existing field untouched
        new_item["embedding_score_raw"] = raw
        new_item["embedding_score_norm"] = norm
        new_item["embedding_rank"] = rank
        reranked.append(new_item)

    return reranked


def rerank_claim_folder(
    claim_folder: str,
    retrieval_filename: str = "retrieval_output.json",
    bm25_filename: str = "bm25_ranked.json",
    output_filename: str = "embedding_ranked.json",
) -> dict:
    """Convenience wrapper for running this stage on ONE claim's output
    folder -- the same outputs/claims/<claim_id>/ folder pipeline.py
    already writes to. claim_folder is supplied by the caller (CLI arg,
    loop variable, batch script, etc.) at run time; nothing here is
    hardcoded to a specific claim.

    Reads:  <claim_folder>/retrieval_output.json  (for english_translation)
            <claim_folder>/bm25_ranked.json        (for ranked_evidence)
    Writes: <claim_folder>/embedding_ranked.json
    """
    folder = Path(claim_folder)

    with open(folder / retrieval_filename, encoding="utf-8") as f:
        retrieval_data = json.load(f)
    with open(folder / bm25_filename, encoding="utf-8") as f:
        bm25_data = json.load(f)

    claim_text_english = retrieval_data["generated_queries"]["english_translation"]
    ranked_evidence = bm25_data["ranked_evidence"]
    claim_meta = bm25_data["claim"]

    reranked = rerank_with_embeddings(claim_text_english, ranked_evidence)

    output = {
        "claim": claim_meta,
        "embedding_model_used": _model_name_used,
        "reranked_evidence": reranked,
    }

    output_path = folder / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[embedding_rerank] Saved {len(reranked)} reranked item(s) -> {output_path}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run embedding-based reranking on one claim's output folder. "
                    "No claim ID or file path is hardcoded -- pass the folder to process."
    )
    parser.add_argument(
        "claim_folder",
        help="Path to a single claim's output folder, e.g. outputs/claims/S2_T_BN_1001 "
             "(must already contain retrieval_output.json and bm25_ranked.json).",
    )
    args = parser.parse_args()

    rerank_claim_folder(args.claim_folder)