"""
Standalone Macro-F1 calculator for FIRE 2026 Subtask 2.

Compares gold labels (from your dev dataset file, e.g. dev_subtask2.json)
against your pipeline's predictions (dev_submission.json), and reports:
  - per-class Precision / Recall / F1 (SUPPORTS, REFUTES)
  - Macro-F1 (the official Subtask 2 metric per the Evaluation page)
  - a confusion matrix
  - any ID mismatches (predictions with no gold label, or vice versa)

USAGE:
    python compute_f1.py --gold data/dev_subtask2.json --pred outputs/dev_submission.json

    # If your gold file uses a different key name for the label
    # (e.g. "Gold_Label" or "Veracity" instead of "Label"):
    python compute_f1.py --gold data/dev_subtask2.json --pred outputs/dev_submission.json --gold-label-key Veracity

    # To evaluate only a specific number of claims (e.g. the first 100
    # by ID, for a quick spot check before the full dev set finishes):
    python compute_f1.py --gold data/dev_subtask2.json --pred outputs/dev_submission.json --limit 100

No third-party dependencies -- only the Python standard library.
"""
import argparse
import json
import sys
from collections import Counter


VALID_LABELS = ("SUPPORTS", "REFUTES")


def load_gold(path: str, label_key: str) -> dict:
    """Returns {claim_id: gold_label}. Raises a clear error immediately
    if the expected label key isn't present, showing what keys ARE
    present on the first record, so you can fix --gold-label-key
    instead of guessing."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print(f"[compute_f1] ERROR: {path} is empty.")
        sys.exit(1)

    first = data[0]
    if label_key not in first:
        print(f"[compute_f1] ERROR: key {label_key!r} not found in gold file records.")
        print(f"[compute_f1] Keys actually present on the first record: {list(first.keys())}")
        print(f"[compute_f1] Re-run with --gold-label-key <correct_key_name>.")
        sys.exit(1)

    gold = {}
    for item in data:
        gold[item["ID"]] = item[label_key].strip().upper()
    return gold


def load_predictions(path: str) -> dict:
    """Returns {claim_id: predicted_label}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    preds = {}
    for item in data:
        preds[item["ID"]] = item["Prediction"].strip().upper()
    return preds


def compute_metrics(gold: dict, preds: dict, limit: int | None = None) -> None:
    gold_ids = set(gold.keys())
    pred_ids = set(preds.keys())

    only_in_gold = gold_ids - pred_ids
    only_in_preds = pred_ids - gold_ids
    common_ids = sorted(gold_ids & pred_ids)  # sorted for reproducibility

    if limit is not None:
        common_ids = common_ids[:limit]

    if not common_ids:
        print("[compute_f1] ERROR: no overlapping claim IDs between gold and predictions. "
              "Check that both files use the same ID format (e.g. 'S2/1627' vs 'S2_1627').")
        sys.exit(1)

    print(f"Evaluating on {len(common_ids)} claim(s) "
          f"(gold has {len(gold_ids)} total, predictions has {len(pred_ids)} total)")
    if only_in_gold:
        print(f"  WARNING: {len(only_in_gold)} gold claim(s) have no prediction yet "
              f"(e.g. still processing) -- excluded from this evaluation: "
              f"{sorted(only_in_gold)[:5]}{' ...' if len(only_in_gold) > 5 else ''}")
    if only_in_preds:
        print(f"  WARNING: {len(only_in_preds)} prediction(s) have no matching gold label "
              f"-- excluded from this evaluation: "
              f"{sorted(only_in_preds)[:5]}{' ...' if len(only_in_preds) > 5 else ''}")
    print()

    # Confusion matrix: confusion[gold_label][pred_label] = count
    confusion = {g: Counter() for g in VALID_LABELS}
    invalid_gold = []
    invalid_pred = []

    for cid in common_ids:
        g = gold[cid]
        p = preds[cid]
        if g not in VALID_LABELS:
            invalid_gold.append((cid, g))
            continue
        if p not in VALID_LABELS:
            invalid_pred.append((cid, p))
            continue
        confusion[g][p] += 1

    if invalid_gold:
        print(f"  WARNING: {len(invalid_gold)} gold label(s) are not exactly SUPPORTS/REFUTES "
              f"(e.g. {invalid_gold[:3]}) -- excluded from metrics.")
    if invalid_pred:
        print(f"  WARNING: {len(invalid_pred)} prediction(s) are not exactly SUPPORTS/REFUTES "
              f"(e.g. {invalid_pred[:3]}) -- excluded from metrics.")
    print()

    # Per-class Precision / Recall / F1
    per_class_f1 = {}
    print(f"{'Label':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for label in VALID_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in VALID_LABELS if g != label)
        fn = sum(confusion[label][p] for p in VALID_LABELS if p != label)
        support = sum(confusion[label].values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_f1[label] = f1

        print(f"{label:<10} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")

    macro_f1 = sum(per_class_f1.values()) / len(per_class_f1)
    total_correct = sum(confusion[g][g] for g in VALID_LABELS)
    total_evaluated = sum(sum(confusion[g].values()) for g in VALID_LABELS)
    accuracy = total_correct / total_evaluated if total_evaluated else 0.0

    print()
    print(f"Accuracy:  {accuracy:.4f}  ({total_correct}/{total_evaluated})")
    print(f"Macro-F1:  {macro_f1:.4f}   <-- official Subtask 2 metric")
    print()

    print("Confusion matrix (rows = gold, columns = predicted):")
    header = "gold\\pred".ljust(12) + "".join(f"{l:>12}" for l in VALID_LABELS)
    print(header)
    for g in VALID_LABELS:
        row = g.ljust(12) + "".join(f"{confusion[g][p]:>12}" for p in VALID_LABELS)
        print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", required=True, help="Path to the gold-labeled dataset file (e.g. data/dev_subtask2.json)")
    parser.add_argument("--pred", required=True, help="Path to your pipeline's predictions file (e.g. outputs/dev_submission.json)")
    parser.add_argument("--gold-label-key", default="Label",
                        help="Key name for the gold label field in --gold (default: 'Label'). "
                             "If the script errors out showing different available keys, re-run with the correct name here.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N overlapping claim IDs (sorted), e.g. --limit 100 for a quick spot check.")
    args = parser.parse_args()

    gold = load_gold(args.gold, args.gold_label_key)
    preds = load_predictions(args.pred)
    compute_metrics(gold, preds, limit=args.limit)


if __name__ == "__main__":
    main()