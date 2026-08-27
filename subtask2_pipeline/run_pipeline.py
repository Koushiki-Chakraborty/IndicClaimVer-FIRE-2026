"""
Single entry point for the FULL Subtask 2 pipeline, end to end.

    python run_pipeline.py

This is a PURE ORCHESTRATOR: it does not contain any retrieval, ranking,
or LLM logic itself -- it only calls the existing stage modules, in the
order their own file dependencies already require, and reports progress.
No existing module's internals are modified or duplicated here.

ACTUAL STAGE ORDER (see note below on why this differs from a naive
1-9 reading of the stage list):

  1. Query Generation   )
  2. Web Search          ) all three happen inside ONE call to
  3. BM25 Ranking        )  pipeline.process_claim() -- that function
                            already bundles them and writes both
                            retrieval_output.json + bm25_ranked.json
  4. Embedding Ranking      (embedding_rerank.rerank_claim_folder)
  5. Entity Extraction      (entity_graph.rank_claim_folder -- extracts
                             AND graph-ranks in one call; needs
                             embedding_ranked.json as input)
  6. Fusion Ranking         (fusion_ranking.rank_claim_folder --
                             needs entity_ranked.json as input)
  7. Full Page Fetch     )  both happen inside ONE call to
  8. Verification (LLM)  )   verification.verify_claim_folder() --
                             it fetches full pages ONLY for the top-3
                             FUSION-ranked items, so fusion must run first
  9. Build Final Submission  (build_submission.py, run once after
                             every claim has been attempted)

WHY THE ORDER DIFFERS FROM A LITERAL 1-9 READING: entity extraction is
only ever invoked from inside entity_graph.py (after embedding
reranking), and full-page fetching is only ever invoked from inside
verification.py (after fusion ranking, and only for the top-3). Making
"Entity Extraction" or "Full Page Fetch" independent, earlier steps
would require changing what those modules read/write -- which the
brief explicitly asked NOT to do. So this script calls each existing
module exactly once, in the only order its actual file dependencies
allow, and simply reports progress at finer granularity than the
function-call granularity (e.g. printing three separate checkmarks for
one process_claim() call, since all three things genuinely did happen
inside it).

RESUMABILITY (pure orchestration, no module logic touched): each stage
is skipped for a claim if that stage's output file already exists,
unless --no-skip-existing is passed. This means an interrupted 1009-claim
run can simply be re-run and it picks up exactly where it left off,
without re-spending Gemini/Tavily API calls on claims already done.

ERROR ISOLATION: if any stage fails for one claim, that claim's error is
logged to config.FAILED_CLAIMS_LOG and the run moves on to the next
claim -- one bad claim never aborts the batch.

VERIFICATION BACKEND (--verifier, default "muril"): stage 7/8 now has two
interchangeable implementations, selected by this flag, both writing the
exact same verification_output.json schema so build_submission.py (and
everything downstream) is unaffected by which one ran:
  - "muril" (default): src/muril_classifier.py's fine-tuned checkpoint,
    classifying the claim against its top-1 fusion-ranked evidence item's
    title+snippet (same convention as src/run_classifier_pipeline.py's
    _top_evidence_text -- reused, not reimplemented). No full-page fetch --
    this is the actual/official path for testing and for the final
    submission. Justification comes from src/gpt_justifier.py (real,
    evidence-grounded explanation of the already-decided Prediction, same
    convention as src/regenerate_classifier_justifications.py) when
    OPENAI_API_KEY is set, falling back to a flat template otherwise so
    this path still runs without that key -- only Prediction requires no
    API key; the richer Justification does.
  - "gemini": the original src/verification.py path -- full-page fetch
    (trafilatura) of the top-3 fusion-ranked items, then a grounded
    google-genai call. Kept available (not deleted) for generating the
    comparison numbers reported in README.md's metrics table.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import config, data_loader
from src.pipeline import process_claim, save_claim_outputs, _sanitize_claim_id
from src.embedding_rerank import rerank_claim_folder as embedding_rerank_claim_folder
from src.entity_graph import rank_claim_folder as entity_rank_claim_folder
from src.fusion_ranking import rank_claim_folder as fusion_rank_claim_folder
from src.verification import verify_claim_folder
from src.run_classifier_pipeline import _top_evidence_text
from src.muril_classifier import VeracityPredictor
from src.gpt_justifier import justify as gpt_justify
from src.neo4j_client import Neo4jClient


def _claim_folder_path(claim_id: str) -> Path:
    """Same sanitization pipeline.py already uses internally -- imported,
    not reimplemented, so there is exactly one place this logic lives."""
    return config.CLAIMS_OUTPUT_DIR / _sanitize_claim_id(claim_id)


def _stage_1_3_retrieval_and_bm25(claim: dict, folder: Path, skip_existing: bool) -> bool:
    """Query Generation + Web Search + BM25 Ranking (one bundled call --
    see module docstring for why these three aren't separable without
    touching pipeline.py's existing logic)."""
    bm25_path = folder / config.BM25_OUTPUT_FILENAME
    if skip_existing and bm25_path.exists():
        print("  - Query Generation   (skipped, already done)")
        print("  - Web Search          (skipped, already done)")
        print("  - BM25 Ranking        (skipped, already done)")
        return True

    result = process_claim(claim, verbose=False)
    save_claim_outputs(claim["id"], result["retrieval_output"], result["bm25_output"])
    print("  \u2713 Query Generation")
    print("  \u2713 Web Search")
    print("  \u2713 BM25 Ranking")
    return True


def _stage_4_embedding(folder: Path, skip_existing: bool) -> bool:
    output_path = folder / "embedding_ranked.json"
    if skip_existing and output_path.exists():
        print("  - Embedding Ranking   (skipped, already done)")
        return True
    embedding_rerank_claim_folder(str(folder))
    print("  \u2713 Embedding Ranking")
    return True


def _stage_5_entity(folder: Path, client: Neo4jClient, skip_existing: bool) -> bool:
    output_path = folder / config.ENTITY_GRAPH_OUTPUT_FILENAME
    if skip_existing and output_path.exists():
        print("  - Entity Extraction   (skipped, already done)")
        return True
    entity_rank_claim_folder(str(folder), client=client)
    print("  \u2713 Entity Extraction")
    return True


def _stage_6_fusion(folder: Path, skip_existing: bool) -> bool:
    output_path = folder / config.FUSION_OUTPUT_FILENAME
    if skip_existing and output_path.exists():
        print("  - Fusion Ranking      (skipped, already done)")
        return True
    fusion_rank_claim_folder(str(folder))
    print("  \u2713 Fusion Ranking")
    return True


def _stage_7_8_verification(folder: Path, skip_existing: bool) -> bool:
    """Full Page Fetch + Verification (LLM) -- one bundled call, see
    module docstring for why full-page fetching can't be pulled out as
    an earlier, independent step without changing verification.py's logic."""
    output_path = folder / config.VERIFICATION_OUTPUT_FILENAME
    if skip_existing and output_path.exists():
        print("  - Full Page Fetch     (skipped, already done)")
        print("  - Verification (LLM)  (skipped, already done)")
        return True
    verify_claim_folder(str(folder))
    print("  \u2713 Full Page Fetch")
    print("  \u2713 Verification (LLM)")
    return True


def _stage_7_8_verification_muril(folder: Path, predictor: VeracityPredictor, skip_existing: bool) -> bool:
    """MuRIL verification -- the default path (see module docstring).
    Classifies the claim against its top-1 fusion-ranked evidence item's
    title+snippet (_top_evidence_text, reused unchanged from
    run_classifier_pipeline.py) -- no full-page fetch, since MuRIL's
    max_length=192 truncates far short of a full article anyway.

    Justification: uses src/gpt_justifier.py to generate a real,
    evidence-grounded explanation of MuRIL's already-decided Prediction --
    same convention src/regenerate_classifier_justifications.py already
    applies to run_classifier_pipeline.py's output, so this path doesn't
    regress to the older flat template. gpt_justifier.justify() never
    reclassifies -- Prediction is fixed input, GPT-4o-mini only explains
    it. Falls back to the flat template if OPENAI_API_KEY isn't set, so
    this stage still runs (degraded, not crashed) without that key --
    same graceful-degradation convention used throughout this codebase
    (NER fallback, translation fallback, Wikidata-linking fallback, etc.).

    Writes the exact same verification_output.json schema the Gemini
    path writes, so build_submission.py works unchanged either way."""
    output_path = folder / config.VERIFICATION_OUTPUT_FILENAME
    if skip_existing and output_path.exists():
        print("  - MuRIL Verification  (skipped, already done)")
        return True

    with open(folder / config.FUSION_OUTPUT_FILENAME, encoding="utf-8") as f:
        fusion_data = json.load(f)

    claim_meta = fusion_data["claim"]
    evidence_text = _top_evidence_text(fusion_data) or " "  # tokenizer needs non-empty text
    label, confidence = predictor.predict_one(claim_meta["text"], evidence_text)

    if config.OPENAI_API_KEY:
        justification = gpt_justify(claim_meta["text"], evidence_text, label)
    else:
        justification = (
            f"Based on the retrieved evidence, the classifier predicts this "
            f"claim {label} with a confidence score of {confidence:.4f}."
        )

    record = {
        "ID": claim_meta["id"],
        "Evidence": evidence_text,
        "Prediction": label,
        "Justification": justification,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print("  \u2713 MuRIL Verification (top-1 evidence, title+snippet)")
    return True


def run_claim(claim: dict, client: Neo4jClient, skip_existing: bool,
              verifier: str = "muril", muril_predictor: VeracityPredictor = None) -> None:
    """Runs every stage for ONE claim, in the real dependency order.
    Raises on the first stage failure -- caller (run_all) is responsible
    for catching it, logging it, and moving on to the next claim."""
    folder = _claim_folder_path(claim["id"])
    _stage_1_3_retrieval_and_bm25(claim, folder, skip_existing)
    _stage_4_embedding(folder, skip_existing)
    _stage_5_entity(folder, client, skip_existing)
    _stage_6_fusion(folder, skip_existing)
    if verifier == "muril":
        _stage_7_8_verification_muril(folder, muril_predictor, skip_existing)
    else:
        _stage_7_8_verification(folder, skip_existing)


def run_all(claims: list, skip_existing: bool = True,
            output_file: Path | None = None, verifier: str = "muril") -> None:
    """Processes every claim in `claims` sequentially, isolating failures
    so one bad claim doesn't stop the other 1008. After every claim has
    been attempted, runs build_submission.py exactly once to aggregate
    all verification_output.json files into the final FIRE submission.

    output_file: OPTIONAL ADD-ON. When None (default -- unchanged
    existing behavior), build_submission.py is invoked with no extra
    flags, so it writes to config.OUTPUT_DIR / config.FINAL_SUBMISSION_FILENAME
    exactly as it always has. When provided (e.g. by --output-file /
    --dataset-file below), it's passed straight through to
    build_submission.py's own --output flag, so this run's aggregation
    is written to a separate file instead. claims_dir is not threaded
    through here separately -- see note in run_all's caller below on
    why config.CLAIMS_OUTPUT_DIR is monkey-patched instead for the
    in-process stages, and passed to build_submission's subprocess call
    explicitly so the two never disagree.

    verifier: "muril" (default) or "gemini" -- see module docstring's
    VERIFICATION BACKEND section. The MuRIL model is loaded exactly once
    here (not per-claim -- loading it 1009 times would be wasteful) and
    shared across every claim in this batch.
    """
    total = len(claims)
    succeeded, failed = 0, []

    client = Neo4jClient()
    client.setup_schema()

    muril_predictor = VeracityPredictor() if verifier == "muril" else None

    try:
        for i, claim in enumerate(claims, start=1):
            claim_id = claim["id"]
            print(f"\n[{i}/{total}] Processing {_sanitize_claim_id(claim_id)}...")
            try:
                run_claim(claim, client, skip_existing, verifier=verifier, muril_predictor=muril_predictor)
                succeeded += 1
            except Exception as e:
                failed.append(claim_id)
                print(f"  \u2717 FAILED: {e}")
                config.OUTPUT_DIR.mkdir(exist_ok=True)
                with open(config.FAILED_CLAIMS_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{claim_id}\t{type(e).__name__}: {e}\n")
                # Deliberately continue to the next claim, per requirement.
    finally:
        client.close()

    print(f"\n{'=' * 70}")
    print(f"ALL CLAIMS PROCESSED: {succeeded} succeeded, {len(failed)} failed (of {total})")
    if failed:
        print(f"Failed claim IDs logged to: {config.FAILED_CLAIMS_LOG}")
    print(f"{'=' * 70}")

    # Final step: build the submission file. Run as a real subprocess
    # (not an imported function call) so build_submission.py's own
    # argparse/CLI logic is used completely unmodified and untouched,
    # exactly as it exists on disk.
    print("\n[run_pipeline] Building final submission file...")
    build_submission_path = Path(__file__).parent / "src" / "build_submission.py"
    build_submission_args = [sys.executable, str(build_submission_path)]
    # config.CLAIMS_OUTPUT_DIR may have been monkey-patched by __main__
    # below (for a --dataset-file run) -- pass the SAME directory
    # explicitly to the subprocess, since a subprocess re-imports
    # config.py fresh and would otherwise see only the original default.
    build_submission_args += ["--claims-dir", str(config.CLAIMS_OUTPUT_DIR)]
    if output_file is not None:
        build_submission_args += ["--output", str(output_file)]
    subprocess.run(build_submission_args, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the FULL Subtask 2 pipeline end-to-end for any number of claims. "
                    "Same claim-selection flags as pipeline.py -- nothing about which "
                    "claims run, or how many, is hardcoded."
    )
    parser.add_argument(
        "--claim-id", action="append", dest="claim_ids", default=None,
        help="Process one specific claim ID. Repeat the flag for more than one.",
    )
    parser.add_argument(
        "--language", default=None, choices=list(config.DATA_FILES.keys()),
        help="Process every claim from one language file.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of claims processed. Omit to process everything selected "
             "(e.g. all 1009 across all 4 languages, if --language/--claim-id are also omitted).",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Re-run every stage even if its output file already exists "
             "(default: skip stages already completed, so a re-run resumes "
             "an interrupted batch instead of starting over).",
    )
    # OPTIONAL ADD-ON: run the exact same pipeline against an arbitrary
    # dataset file (e.g. dev_subtask2.json) instead of the 4 built-in
    # test language files, writing to separate output locations so the
    # existing test-set run (outputs/claims/, outputs/final_submission.json)
    # is never touched or overwritten. All three flags are optional and
    # unused by default -- omitting --dataset-file means every line below
    # is skipped and this script behaves EXACTLY as it did before this
    # add-on existed.
    parser.add_argument(
        "--dataset-file", default=None,
        help="Path to an arbitrary claims JSON file (e.g. data/dev_subtask2.json), "
             "used INSTEAD of --language/--claim-id/config.DATA_FILES. Same "
             "[{'ID':..., 'Text':...}, ...] input schema as the existing test files.",
    )
    parser.add_argument(
        "--dataset-lang", default="dev",
        help="Language tag to attach to every claim loaded via --dataset-file "
             "(default: 'dev'). Only used together with --dataset-file.",
    )
    parser.add_argument(
        "--claims-dir", default=None,
        help="Directory to write per-claim folders to for this run. "
             "Default when --dataset-file is given: outputs/claims_dev "
             "(so it never collides with the existing outputs/claims). "
             "Default otherwise: outputs/claims (unchanged existing behavior).",
    )
    parser.add_argument(
        "--output-file", default=None,
        help="Path for the aggregated submission JSON for this run. "
             "Default when --dataset-file is given: outputs/dev_submission.json. "
             "Default otherwise: outputs/final_submission.json (unchanged existing behavior).",
    )
    parser.add_argument(
        "--verifier", default="muril", choices=["muril", "gemini"],
        help="Veracity-prediction backend for stage 7/8 (default: muril -- the "
             "reproducible, no-API-quota path used for testing and the actual "
             "final submission). 'gemini' runs the original full-page-fetch + "
             "grounded LLM path (src/verification.py) instead, kept available "
             "for regenerating the Gemini comparison numbers in README.md.",
    )
    args = parser.parse_args()

    if args.dataset_file:
        # Load claims from the arbitrary dataset file instead of the 4
        # built-in test language files -- data_loader.select_claims()
        # (and everything it reads from config.DATA_FILES) is untouched.
        selected_claims = data_loader.load_claims_from_file(args.dataset_file, lang=args.dataset_lang)

        # Route this run's per-claim folders and final output to SEPARATE
        # locations by default, so a dev run can never overwrite the
        # existing test-set outputs. config.CLAIMS_OUTPUT_DIR is a plain
        # module attribute read fresh (not cached) by every in-process
        # stage (pipeline.py's save_claim_outputs, this script's own
        # _claim_folder_path), so reassigning it here is sufficient --
        # no stage's internal logic is modified.
        claims_dir = Path(args.claims_dir) if args.claims_dir else (config.OUTPUT_DIR / "claims_dev")
        config.CLAIMS_OUTPUT_DIR = claims_dir

        output_file = Path(args.output_file) if args.output_file else (config.OUTPUT_DIR / "dev_submission.json")

        print(f"[run_pipeline] Using dataset file: {args.dataset_file}")
        print(f"[run_pipeline] Per-claim outputs -> {claims_dir}")
        print(f"[run_pipeline] Final submission   -> {output_file}")
    else:
        # UNCHANGED existing behavior -- exactly as before this add-on.
        selected_claims = data_loader.select_claims(
            claim_ids=args.claim_ids, language=args.language, limit=args.limit,
        )
        output_file = None  # build_submission.py uses its own defaults, as before

    if args.dataset_file and args.limit is not None:
        selected_claims = selected_claims[:args.limit]

    print(f"[run_pipeline] Selected {len(selected_claims)} claim(s) to process.")
    print(f"[run_pipeline] Verification backend: {args.verifier}"
          f"{' (default -- no GEMINI_API_KEY needed)' if args.verifier == 'muril' else ''}")

    run_all(selected_claims, skip_existing=not args.no_skip_existing,
            output_file=output_file, verifier=args.verifier)

    final_path = output_file if output_file is not None else (config.OUTPUT_DIR / config.FINAL_SUBMISSION_FILENAME)
    print(f"\n[run_pipeline] Done. Final submission -> {final_path}")