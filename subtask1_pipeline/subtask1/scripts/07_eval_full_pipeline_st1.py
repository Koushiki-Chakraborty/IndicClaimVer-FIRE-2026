"""Full dev-set pipeline: KG-boosted top-3 retrieval + MuRIL veracity
classification (config.MURIL_CLASSIFIER_MODEL, loaded straight from the
Hub via subtask1/muril_classifier.py -- root README.md, Notes -> "Classifier").
There is no training code here; this script only runs the
already fine-tuned checkpoint.

Writes the same three files subtask1/results/devResult/kg_classifier/
already carried from an external run (final_predictions.json,
submission.json, evaluation_metrics.json), so this script can now
actually reproduce that folder's contents end to end instead of it only
being buildable by hand outside the repo.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from subtask1 import data_loader
from subtask1.evaluate import (
    evaluate_retrieval, evaluate_veracity, format_submission_row, write_submission,
)
from subtask1.muril_classifier import VeracityPredictor
from subtask1.neo4j_client import Neo4jClient
from subtask1.retriever import DenseRetriever, KGBoostedRetriever

RESULTS_DIR = Path(__file__).parent.parent / "results" / "devResult" / "kg_classifier"


def per_class_metrics(golds: list, preds: list) -> dict:
    """Same fields/shape as the existing evaluation_metrics.json in this
    folder (built externally from the notebook run) -- accuracy, macro
    F1, and per-class precision/recall/F1/support for SUPPORTS/REFUTES."""
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


if __name__ == "__main__":
    _train, dev, pool = data_loader.load_all()
    if not (dev and pool):
        print("Need dev_subtask1.json and evidence_pool.json in data/. Aborting.")
        sys.exit(1)

    pool_text_to_id = data_loader.build_pool_text_to_id(pool)
    pool_id_to_evidence = {item["ID"]: item["Evidence"] for item in pool}

    client = Neo4jClient()
    dense = DenseRetriever(client)
    kg_retriever = KGBoostedRetriever(client, dense)

    # ---- Stage 1: KG-boosted top-3 retrieval, per claim (each call hits Neo4j) ----
    retrieval_results, top3_by_claim = [], []
    for item in dev:
        top3 = kg_retriever.retrieve(item["ID"], item["Text"], top_k=3)
        if len(top3) < 3:
            # Spec disallows anything but exactly 3 -- pad from a wider
            # Dense pull if the language pool itself was too thin for the
            # KG-boosted top-3 to fill.
            padding = [eid for eid, _ in dense.retrieve(item["Text"], top_k=10, return_scores=True)
                       if eid not in top3]
            top3 = (top3 + padding)[:3]
        retrieval_results.append({"claim_item": item, "retrieved_ids": top3})
        top3_by_claim.append(top3)
    client.close()

    # ---- Stage 2: veracity classification, one batched call over every
    # (claim, evidence) pair across all 3 retrieved evidence items per
    # claim -- not just top-1 -- so final_predictions.json carries the
    # same per-evidence breakdown as the existing notebook-built file. ----
    predictor = VeracityPredictor()

    pair_claim_ids, pair_evidence_ids, pair_claim_texts, pair_evidence_texts = [], [], [], []
    for item, top3 in zip(dev, top3_by_claim):
        for eid in top3:
            pair_claim_ids.append(item["ID"])
            pair_evidence_ids.append(eid)
            pair_claim_texts.append(item["Text"])
            pair_evidence_texts.append(pool_id_to_evidence.get(eid, "") or " ")

    print(f"\nClassifying {len(pair_claim_texts)} (claim, evidence) pairs "
          f"({len(dev)} claims x up to 3 evidence each)...")
    pair_predictions = predictor.predict_batch(pair_claim_texts, pair_evidence_texts)

    detailed_rows = []
    preds_by_claim = {}  # claim_id -> [(evidence_id, label, confidence), ...], retrieval order preserved
    for claim_id, evidence_id, claim_text, evidence_text, (label, confidence) in zip(
        pair_claim_ids, pair_evidence_ids, pair_claim_texts, pair_evidence_texts, pair_predictions
    ):
        preds_by_claim.setdefault(claim_id, []).append((evidence_id, label, confidence))

    for item, top3 in zip(dev, top3_by_claim):
        claim_id = item["ID"]
        for idx, (evidence_id, label, confidence) in enumerate(preds_by_claim[claim_id], start=1):
            detailed_rows.append({
                "ID": f"{claim_id}_{idx:02d}",
                "Claim_ID": claim_id,
                "Evidence_ID": evidence_id,
                "Claim": item["Text"],
                "Evidence": pool_id_to_evidence.get(evidence_id, ""),
                "Retrieval_Score": None,  # KGBoostedRetriever.retrieve() returns ranked IDs, not scores
                "Predicted_Label": label,
                "Confidence": round(confidence, 4),
            })

    # Final per-claim Prediction = the top-1 (most relevant) evidence's
    # label -- same convention results/devResult/make_submission.py
    # already uses to collapse the 3-rows-per-claim detailed file down
    # to one prediction per claim.
    veracity_preds = [preds_by_claim[item["ID"]][0][1] for item in dev]
    veracity_golds = [item["Label"] for item in dev]

    output_rows = [
        format_submission_row(item["ID"], top3, pool_id_to_evidence, pred)
        for item, top3, pred in zip(dev, top3_by_claim, veracity_preds)
    ]

    retrieval_metrics = evaluate_retrieval(retrieval_results, pool_text_to_id, k=3)
    veracity_metrics = evaluate_veracity(veracity_preds, veracity_golds)
    detailed_metrics = per_class_metrics(veracity_golds, veracity_preds)

    print("\n=== Retrieval metrics (dev) ===")
    print(retrieval_metrics)
    print("\n=== Veracity metrics (dev, MuRIL classifier via HuggingFace) ===")
    print(f"Accuracy: {detailed_metrics['accuracy']:.4f}  macro_F1: {veracity_metrics['macro_F1']:.4f}")
    print(f"Confusion matrix {veracity_metrics['confusion_matrix_labels']}: "
          f"{veracity_metrics['confusion_matrix']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    detailed_path = RESULTS_DIR / "final_predictions.json"
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_rows, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(detailed_rows)} detailed rows -> {detailed_path}")

    metrics_path = RESULTS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(detailed_metrics, f, ensure_ascii=False, indent=2)
    print(f"Wrote evaluation metrics -> {metrics_path}")

    write_submission(output_rows, str(RESULTS_DIR / "submission.json"),
                      expected_ids=[item["ID"] for item in dev])
