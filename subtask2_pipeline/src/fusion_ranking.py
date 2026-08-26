"""
STEP 8: Reciprocal Rank Fusion (RRF).

Single responsibility: given the entity-ranked evidence list (output of
entity_graph_kg.py / entity_graph.py), combine the three RANKS already
present on every item -- bm25_rank, embedding_rank, entity_rank -- into
one fusion score using the standard RRF formula:

    RRF(d) = sum_i  1 / (k + r_i(d))

where r_i(d) is document d's rank under signal i (BM25, embedding,
entity overlap), and k = 60 (the standard constant from the original
RRF paper -- Cormack, Clarke & Buettcher 2009 -- widely reused as a
default because it works well across a broad range of rank
distributions without per-dataset tuning).

WHY RANKS AND NOT RAW SCORES: bm25_score_raw, embedding_score_raw, and
entity_overlap_raw live on completely different, incomparable scales
(BM25 scores are unbounded corpus-dependent numbers, embedding
similarity is roughly 0-1, entity overlap is a small integer count).
Averaging or weighting raw scores directly would let whichever signal
happens to have the largest numeric range silently dominate the fusion.
Ranks are already scale-free -- "1st place" means the same thing
regardless of which signal produced it -- which is exactly what makes
RRF robust to near-identical or oddly-distributed raw scores: two
documents with almost equal BM25 scores that happen to land at rank 3
and rank 4 contribute predictably different amounts (1/(60+3) vs
1/(60+4)) regardless of how close or far apart their raw scores were.

No new dependencies -- this is pure arithmetic over fields that already
exist on every evidence item. Does not modify BM25, embedding, or
entity graph fields -- only appends fusion_score and fusion_rank,
exactly like every prior stage appends its own fields and nothing else.
"""
import json
from pathlib import Path

# Standard RRF constant. Keeping documents with poor early ranks from
# contributing near-zero (or wildly different) amounts is the whole
# point of adding k in the denominator instead of just summing 1/rank --
# k=60 is the conventional default used across IR literature and
# large-scale search fusion systems (e.g. Elasticsearch's built-in RRF).
RRF_K = 60


def compute_rrf_score(ranks: list, k: int = RRF_K) -> float:
    """ranks: a list of rank integers (1 = best) from each signal, e.g.
    [bm25_rank, embedding_rank, entity_rank]. Missing/None ranks are
    skipped rather than crashing the whole computation -- a document
    that's absent from one signal simply doesn't get a contribution
    from that signal, instead of poisoning the sum with a bad value.

    Returns the summed RRF contribution across all provided ranks.
    """
    total = 0.0
    for r in ranks:
        if r is None:
            continue
        total += 1.0 / (k + r)
    return total


def rank_by_rrf(ranked_evidence: list, k: int = RRF_K) -> list:
    """Pure, dynamic fusion function -- no file I/O, no hardcoded
    claim/path. Caller supplies the data.

    ranked_evidence: the "entity_ranked_evidence" list from
    entity_graph_kg.py's / entity_graph.py's output -- each item
    already has title/url/snippet/bm25_*/embedding_*/entity_* fields.

    Returns a NEW list, sorted by fusion_score descending, with
    fusion_score / fusion_rank added to every item. All existing
    fields are preserved unchanged.
    """
    if not ranked_evidence:
        return []

    scored = []
    for item in ranked_evidence:
        ranks = [item.get("bm25_rank"), item.get("embedding_rank"), item.get("entity_rank")]
        fusion_score = compute_rrf_score(ranks, k=k)
        scored.append((item, fusion_score))

    # Sort by fusion_score descending. Ties (possible if two items land
    # on identical ranks across all three signals) keep their relative
    # input order, since Python's sort is stable.
    scored.sort(key=lambda pair: -pair[1])

    fused = []
    for rank, (item, fusion_score) in enumerate(scored, start=1):
        new_item = dict(item)  # copy -- preserves every existing field untouched
        new_item["fusion_score"] = fusion_score
        new_item["fusion_rank"] = rank
        fused.append(new_item)

    return fused


def rank_claim_folder(
    claim_folder: str,
    entity_filename: str = "entity_ranked.json",
    output_filename: str = "fusion_ranked.json",
    k: int = RRF_K,
) -> dict:
    """Convenience wrapper for running this stage on ONE claim's output
    folder -- the same outputs/claims/<claim_id>/ folder every prior
    stage already writes to. claim_folder is supplied by the caller
    (CLI arg, loop variable, batch script, etc.) at run time; nothing
    here is hardcoded to a specific claim.

    Reads:  <claim_folder>/entity_ranked.json    (for entity_ranked_evidence)
    Writes: <claim_folder>/fusion_ranked.json
    """
    folder = Path(claim_folder)

    with open(folder / entity_filename, encoding="utf-8") as f:
        entity_data = json.load(f)

    claim_meta = entity_data["claim"]
    entity_ranked_evidence = entity_data["entity_ranked_evidence"]

    fused = rank_by_rrf(entity_ranked_evidence, k=k)

    output = {"claim": claim_meta, "fusion_ranked_evidence": fused}

    output_path = folder / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[fusion_ranking] Saved {len(fused)} fusion-ranked item(s) -> {output_path}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Reciprocal Rank Fusion on one claim's output folder. "
                    "No claim ID or file path is hardcoded -- pass the folder to process."
    )
    parser.add_argument(
        "claim_folder",
        help="Path to a single claim's output folder, e.g. outputs/claims/S2_T_BN_1001 "
             "(must already contain entity_ranked.json).",
    )
    parser.add_argument(
        "--k", type=int, default=RRF_K,
        help=f"RRF constant (default: {RRF_K}, the standard value).",
    )
    args = parser.parse_args()

    rank_claim_folder(args.claim_folder, k=args.k)