"""
Orchestrates the Subtask 2 pipeline through the BM25 ranking stage:

  1. Query generation (native + English)
  2. Text retrieval (search API, both queries)
  3. Merge + deduplicate
  4. BM25 ranking (lexical only -- see bm25_ranker.py)

Entity extraction, entity overlap, fusion ranking, embeddings, and
Knowledge Graph stages are NOT part of this pipeline -- deliberately
deferred to later.

This module is fully dynamic: it contains no hardcoded claim IDs,
filenames, or language names. Claim selection lives entirely in
data_loader.select_claims(); this file just processes whatever list of
claims it's handed -- one claim, a hand-picked few, or the whole
dataset, with no branching on how many there are.

pipeline.py's only two jobs are: (1) orchestrate the per-claim steps in
order, and (2) centrally save each claim's two output files. No other
module writes to disk.
"""
import json

from . import config, data_loader, query_generation, retrieval, merge_dedup, bm25

# Optional add-on -- see src/evidence_filter.py docstring. Import failure
# (should not normally happen -- no external dependency) degrades to
# skipping this stage entirely, i.e. identical behavior to before this
# add-on existed, rather than crashing pipeline.py.
try:
    from . import evidence_filter
    _FILTERING_AVAILABLE = True
except Exception as _e:
    print(f"[pipeline] WARNING: evidence_filter unavailable ({_e}); "
          f"skipping evidence filtering stage.")
    evidence_filter = None
    _FILTERING_AVAILABLE = False


def process_claim(claim: dict, verbose: bool = True) -> dict:
    """claim: {"id": str, "text": str, "lang": str}

    Runs query generation -> retrieval -> merge/dedup -> BM25 ranking
    for exactly this one claim and returns the two payloads that will
    be written to disk for it:

        {"retrieval_output": {...}, "bm25_output": {...}}
    """
    claim_id, claim_text, lang = claim["id"], claim["text"], claim["lang"]

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"CLAIM [{claim_id}] ({lang})")
        print(f"{claim_text}")
        print(f"{'=' * 70}")

    # STEP 1: Query generation
    if verbose:
        print("\n[query_generation] Generating queries...")
    queries = query_generation.generate_queries(claim_text, lang)
    if verbose:
        print(f"  native_query : {queries['native_query']}")
        print(f"  english_query: {queries['english_query']}")
        print(f"  translation  : {queries['english_translation']}")

    # STEP 2: Retrieval (both queries)
    if verbose:
        print("\n[retrieval] Retrieving from search API...")
    results_native = retrieval.search(queries["native_query"])
    results_english = retrieval.search(queries["english_query"])
    if verbose:
        print(f"  native query  -> {len(results_native)} results")
        print(f"  english query -> {len(results_english)} results")

    # STEP 3: Merge + dedup
    merged = merge_dedup.merge_and_dedup(results_native, results_english)
    if verbose:
        print(f"\n[merge_dedup] Merged + deduplicated -> {len(merged)} unique results")

    # STEP 3b (OPTIONAL ADD-ON, 2026-07): Evidence filtering -- removes
    # obvious non-article noise (cookie/ToS/login pages, empty snippets)
    # before ranking. See src/evidence_filter.py docstring. Strict subset
    # of `merged`; falls back to `merged` unchanged if filtering is
    # unavailable or would remove everything. `retrieval_output` below
    # still records the original unfiltered `merged` pool (so nothing
    # about the existing JSON schema changes) -- only what's fed into
    # BM25 ranking is affected.
    pool_for_ranking = merged
    if _FILTERING_AVAILABLE:
        pool_for_ranking = evidence_filter.filter_evidence(merged)
        if verbose:
            print(f"[evidence_filter] {len(merged)} -> {len(pool_for_ranking)} "
                  f"item(s) after filtering")

    # STEP 4: BM25 ranking (lexical only)
    # bm25.py has one responsibility: score a single query string against
    # the evidence pool. We combine both generated queries into one
    # string here so a single ranking pass covers evidence in either
    # language, without bm25.py needing any script/language awareness.
    if verbose:
        print("\n[bm25] Ranking evidence (BM25 only)...")
    combined_query = f"{queries['native_query']} {queries['english_query']}"
    ranked = bm25.rank_evidence(
        query=combined_query,
        evidence_pool=pool_for_ranking,
    )
    if verbose:
        print(f"\n[bm25] Ranked {len(ranked)} evidence item(s):")
        for r in ranked[:5]:
            print(f"  #{r['bm25_rank']}  norm={r['bm25_score_norm']:.2f}  {r['title']}")

    retrieval_output = {
        "claim": {"id": claim_id, "text": claim_text, "lang": lang},
        "generated_queries": queries,
        "native_retrieval": results_native,
        "english_retrieval": results_english,
        "merged_results": merged,
    }

    bm25_output = {
        "claim": {"id": claim_id, "text": claim_text, "lang": lang},
        "ranked_evidence": ranked,
    }

    return {"retrieval_output": retrieval_output, "bm25_output": bm25_output}


def _sanitize_claim_id(claim_id: str) -> str:
    """Claim IDs contain '/' (e.g. 'S2/T/BN/1001'), which isn't a valid
    path component -- swap to '_' for the output folder name only. This
    is a filesystem-safety transform, not a data change."""
    return claim_id.replace("/", "_")


def save_claim_outputs(claim_id: str, retrieval_output: dict, bm25_output: dict) -> None:
    """The ONLY place in the pipeline that writes files (per Objective 3
    -- individual stage modules never save to disk themselves). Writes
    both output files for one claim under outputs/claims/<safe_id>/."""
    claim_dir = config.CLAIMS_OUTPUT_DIR / _sanitize_claim_id(claim_id)
    claim_dir.mkdir(parents=True, exist_ok=True)

    with open(claim_dir / config.RETRIEVAL_OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(retrieval_output, f, ensure_ascii=False, indent=2)

    with open(claim_dir / config.BM25_OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(bm25_output, f, ensure_ascii=False, indent=2)


def run(claims: list) -> list:
    """Processes + saves outputs for any list of claims. Identical code
    path for 1 claim, a hand-picked list, or the entire dataset -- the
    only thing that changes is what data_loader.select_claims() handed
    us before this was called.

    A single claim erroring out (bad API response, etc.) is logged to
    config.FAILED_CLAIMS_LOG and skipped, so one failure doesn't abort
    a large batch run.
    """
    results = []
    for claim in claims:
        try:
            result = process_claim(claim)
            save_claim_outputs(claim["id"], result["retrieval_output"], result["bm25_output"])
            results.append(result)
        except Exception as e:
            print(f"[pipeline] ERROR processing claim {claim.get('id')}: {e}")
            config.OUTPUT_DIR.mkdir(exist_ok=True)
            with open(config.FAILED_CLAIMS_LOG, "a", encoding="utf-8") as f:
                f.write(f"{claim.get('id')}\t{e}\n")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the retrieval + BM25 pipeline on any number of claims. "
                    "Nothing about WHICH claims run is hardcoded -- select entirely "
                    "via these flags."
    )
    parser.add_argument(
        "--claim-id", action="append", dest="claim_ids", default=None,
        help="Process one specific claim ID. Repeat the flag for more than one, "
             "e.g. --claim-id S2/T/BN/1001 --claim-id S2/T/HI/2001",
    )
    parser.add_argument(
        "--language", default=None, choices=list(config.DATA_FILES.keys()),
        help="Process every claim from one language file.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of claims processed, applied after --language/--claim-id "
             "selection. Omit to process everything selected.",
    )
    args = parser.parse_args()

    selected = data_loader.select_claims(
        claim_ids=args.claim_ids, language=args.language, limit=args.limit,
    )
    print(f"[pipeline] Selected {len(selected)} claim(s) to process.")

    run(selected)

    print(f"\n[pipeline] Done. Outputs written under {config.CLAIMS_OUTPUT_DIR}")