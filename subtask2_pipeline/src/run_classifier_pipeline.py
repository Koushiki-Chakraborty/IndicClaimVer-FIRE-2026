"""
Standalone MuRIL veracity-classification pipeline for Subtask 2.

Runs AFTER run_pipeline.py's retrieval/ranking stages have already
produced fusion_ranked.json for every claim -- this script reads that
file only; it does no retrieval, ranking, or web/API calls of its own.
For each claim, classifies the top-1 fusion-ranked evidence item
against the claim text using the HuggingFace MuRIL checkpoint
(src/muril_classifier.py -- the SAME checkpoint subtask1_pipeline uses,
reused as-is here, not retrained on any Subtask 2 data -- see root README.md,
Notes -> "Classifier").

This is a separate, standalone prediction path from run_pipeline.py's
Gemini-based verification.py -- the two are never merged into one
default submission (same relationship as Subtask 1's dense_gpt vs
kg_classifier paths).

Two modes:

  --mode dev   Evaluate against data/dev_subtask2.json's gold "Label"
               field. Writes:
                 output/classifier_output/dev_submission.json
                 output/classifier_output/dev_metrics.json
               dev_metrics.json is computed at the SAME granularity as
               output/GPT_output/dev_metrics.json -- one row per claim,
               n=631. The dev_metrics.json this script overwrites was
               previously computed per (claim, evidence) triple
               (n=1891, ~3x inflated relative to the 631 actual dev
               claims), which made it not directly comparable to
               GPT_output's per-claim n=631 number despite both being
               shown side by side in the root README.md's "Results"
               table. This script fixes that going forward -- both
               rows are now on the same basis.

  --mode test  No gold labels available. Splits predictions by each
               claim's own `lang` tag (read from fusion_ranked.json,
               set upstream by data_loader.py) into the 4 official
               submission files:
                 output/classifier_output/language_outputs/{English,Hindi,Bengali,CodeMix}_subtask2.json

Usage:
    python -m src.run_classifier_pipeline --mode dev
    python -m src.run_classifier_pipeline --mode test
    python -m src.run_classifier_pipeline --mode dev --limit 20   # quick smoke test
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.muril_classifier import VeracityPredictor

LANG_TO_FILENAME = {
    "english": "English_subtask2.json",
    "hindi": "Hindi_subtask2.json",
    "bengali": "Bengali_subtask2.json",
    "codemix": "CodeMix_subtask2.json",
}

DEV_DATASET_FILE = config.DATA_DIR / "dev_subtask2.json"
DEV_CLAIMS_DIR_DEFAULT = config.OUTPUT_DIR / "claims_dev"
TEST_CLAIMS_DIR_DEFAULT = config.CLAIMS_OUTPUT_DIR  # output/claims
CLASSIFIER_OUTPUT_DIR = config.OUTPUT_DIR / "classifier_output"


def _load_gold_labels(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["ID"]: item["Label"].strip().upper() for item in data}


def _top_evidence_text(fusion_data: dict) -> str:
    """Returns the top fusion-ranked item's evidence text, or "" if this
    claim has no ranked evidence at all (retrieval genuinely found
    nothing). Uses title+snippet -- the same "evidence text" convention
    already used everywhere else in this pipeline (bm25.py,
    embedding_rerank.py, entity_graph.py all score/compare on
    f"{title} {snippet}") -- not a fresh full-article fetch, since
    MuRIL's max_length=192 truncates far short of a full article
    anyway, and this avoids re-fetching every URL verification.py
    already fetched full text for on the Gemini path."""
    ranked = fusion_data.get("fusion_ranked_evidence", [])
    if not ranked:
        return ""
    top = sorted(ranked, key=lambda x: x["fusion_rank"])[0]
    title, snippet = top.get("title", ""), top.get("snippet", "")
    return f"{title} {snippet}".strip()


def _classify_claims_dir(claims_dir: Path, predictor: VeracityPredictor, limit: int = None) -> list:
    """Reads every claim folder's fusion_ranked.json, classifies claim
    vs. its top-1 evidence, and returns a list of
    {"ID", "Claim_lang", "Evidence", "Prediction", "Justification"} dicts.
    Claim_lang is used by --mode test to split into per-language files
    and is dropped before writing the final record (not part of the
    official schema)."""
    folders = sorted(p for p in claims_dir.iterdir() if p.is_dir())
    if limit is not None:
        folders = folders[:limit]

    claim_ids, claim_langs, claim_texts, evidence_texts = [], [], [], []
    skipped = []
    for folder in folders:
        fusion_path = folder / config.FUSION_OUTPUT_FILENAME
        if not fusion_path.exists():
            skipped.append(folder.name)
            continue
        with open(fusion_path, encoding="utf-8") as f:
            fusion_data = json.load(f)
        claim_meta = fusion_data["claim"]
        evidence_text = _top_evidence_text(fusion_data)
        claim_ids.append(claim_meta["id"])
        claim_langs.append(claim_meta.get("lang", ""))
        claim_texts.append(claim_meta["text"])
        evidence_texts.append(evidence_text or " ")  # tokenizer needs non-empty text

    if skipped:
        print(f"[run_classifier_pipeline] WARNING: {len(skipped)} claim folder(s) have no "
              f"{config.FUSION_OUTPUT_FILENAME} yet (run_pipeline.py hasn't reached them), "
              f"skipped: {skipped[:10]}{' ...' if len(skipped) > 10 else ''}")

    print(f"[run_classifier_pipeline] Classifying {len(claim_ids)} claim(s) against their "
          f"top-1 fusion-ranked evidence...")
    predictions = predictor.predict_batch(claim_texts, evidence_texts)

    records = []
    for claim_id, lang, evidence_text, (label, confidence) in zip(
        claim_ids, claim_langs, evidence_texts, predictions
    ):
        records.append({
            "ID": claim_id,
            "Claim_lang": lang,
            "Evidence": evidence_text,
            "Prediction": label,
            "Justification": (
                f"Based on the retrieved evidence, the classifier predicts this "
                f"claim {label} with a confidence score of {confidence:.4f}."
            ),
        })
    return records


def _per_class_metrics(golds: list, preds: list) -> dict:
    """Same schema as subtask1_pipeline's
    kg_classifier/evaluation_metrics.json -- accuracy, macro F1,
    per-class precision/recall/F1/support."""
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    labels = ["SUPPORTS", "REFUTES"]
    accuracy = accuracy_score(golds, preds)
    f1_macro = f1_score(golds, preds, average="macro", labels=labels, zero_division=0)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        golds, preds, labels=labels, zero_division=0
    )
    matches = sum(1 for g, p in zip(golds, preds) if g == p)
    return {
        "total_compared": len(golds),
        "matches": matches,
        "mismatches": len(golds) - matches,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision_supports": precision[0],
        "recall_supports": recall[0],
        "f1_supports": f1_per_class[0],
        "support_supports": int(support[0]),
        "precision_refutes": precision[1],
        "recall_refutes": recall[1],
        "f1_refutes": f1_per_class[1],
        "support_refutes": int(support[1]),
    }


def run_dev(claims_dir: Path, limit: int = None) -> None:
    if not DEV_DATASET_FILE.exists():
        raise FileNotFoundError(f"{DEV_DATASET_FILE} not found.")
    gold = _load_gold_labels(DEV_DATASET_FILE)

    predictor = VeracityPredictor()
    records = _classify_claims_dir(claims_dir, predictor, limit=limit)

    golds, preds = [], []
    output_records = []
    for r in records:
        cid = r["ID"]
        if cid not in gold:
            print(f"[run_classifier_pipeline] WARNING: {cid} has no gold label in "
                  f"{DEV_DATASET_FILE.name}, excluded from metrics (still included in dev_submission.json).")
        else:
            golds.append(gold[cid])
            preds.append(r["Prediction"])
        output_records.append({k: v for k, v in r.items() if k != "Claim_lang"})

    CLASSIFIER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = CLASSIFIER_OUTPUT_DIR / "dev_submission.json"
    with open(submission_path, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)
    print(f"[run_classifier_pipeline] Wrote {len(output_records)} record(s) -> {submission_path}")

    metrics = _per_class_metrics(golds, preds)
    metrics_path = CLASSIFIER_OUTPUT_DIR / "dev_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[run_classifier_pipeline] Wrote metrics (n={metrics['total_compared']}, "
          f"accuracy={metrics['accuracy']:.4f}, macro_F1={metrics['f1_macro']:.4f}) -> {metrics_path}")


def run_test(claims_dir: Path, limit: int = None) -> None:
    predictor = VeracityPredictor()
    records = _classify_claims_dir(claims_dir, predictor, limit=limit)

    by_lang = {lang: [] for lang in LANG_TO_FILENAME}
    unknown_lang = []
    for r in records:
        lang = (r.get("Claim_lang") or "").lower().strip()
        output_record = {k: v for k, v in r.items() if k != "Claim_lang"}
        if lang in by_lang:
            by_lang[lang].append(output_record)
        else:
            unknown_lang.append(r["ID"])
            by_lang.setdefault("english", []).append(output_record)  # best-effort bucket, never dropped

    if unknown_lang:
        print(f"[run_classifier_pipeline] WARNING: {len(unknown_lang)} claim(s) had an "
              f"unrecognized lang tag, bucketed into English as a fallback: {unknown_lang[:10]}")

    out_dir = CLASSIFIER_OUTPUT_DIR / "language_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang, filename in LANG_TO_FILENAME.items():
        out_path = out_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(by_lang[lang], f, ensure_ascii=False, indent=2)
        print(f"[run_classifier_pipeline] Wrote {len(by_lang[lang])} record(s) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the standalone MuRIL (HuggingFace) veracity classifier over "
                    "already-computed fusion_ranked.json files -- no retrieval/ranking/API "
                    "calls of its own; run run_pipeline.py's earlier stages first."
    )
    parser.add_argument("--mode", choices=["dev", "test"], required=True,
                        help="'dev': evaluate against data/dev_subtask2.json gold labels. "
                             "'test': split predictions into the 4 official per-language submission files.")
    parser.add_argument("--claims-dir", default=None,
                        help="Directory of per-claim folders to read fusion_ranked.json from. "
                             "Default: output/claims_dev for --mode dev, output/claims for --mode test.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N claim folders (sorted), for a quick smoke test.")
    args = parser.parse_args()

    if args.mode == "dev":
        claims_dir = Path(args.claims_dir) if args.claims_dir else DEV_CLAIMS_DIR_DEFAULT
        if not claims_dir.exists():
            raise FileNotFoundError(
                f"{claims_dir} not found -- run `python run_pipeline.py --dataset-file "
                f"data/dev_subtask2.json --dataset-lang dev` first to populate fusion_ranked.json "
                f"for every dev claim."
            )
        run_dev(claims_dir, limit=args.limit)
    else:
        claims_dir = Path(args.claims_dir) if args.claims_dir else TEST_CLAIMS_DIR_DEFAULT
        if not claims_dir.exists():
            raise FileNotFoundError(
                f"{claims_dir} not found -- run `python run_pipeline.py` first to populate "
                f"fusion_ranked.json for every test claim."
            )
        run_test(claims_dir, limit=args.limit)
