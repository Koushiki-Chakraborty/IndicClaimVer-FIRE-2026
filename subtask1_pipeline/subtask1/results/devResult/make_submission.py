"""
Generates submission.json by combining:
  - kg/result.json        → top_3_evidence (array of evidence text strings)
  - kg_classifier/final_predictions.json → Predicted_Label (one label per claim)

Output format:
[
  {
    "ID": "S1/1719",
    "top_3_evidence": ["ev1", "ev2", "ev3"],
    "Prediction": "SUPPORTS"
  },
  ...
]
"""

import json
import os

BASE = os.path.dirname(__file__)

KG_RESULT      = os.path.join(BASE, "kg", "result.json")
PREDICTIONS    = os.path.join(BASE, "kg_classifier", "final_predictions.json")
OUTPUT         = os.path.join(BASE, "kg_classifier", "submission.json")

print("Loading kg/result.json …")
with open(KG_RESULT, encoding="utf-8") as f:
    kg_results = json.load(f)

print("Loading kg_classifier/final_predictions.json …")
with open(PREDICTIONS, encoding="utf-8") as f:
    predictions = json.load(f)

# Build a map: Claim_ID -> Predicted_Label
# Each claim appears 3 times (_01, _02, _03); all carry the same label.
# We take whichever we see first (lowest index due to JSON array order).
pred_map = {}
for row in predictions:
    cid = row["Claim_ID"]
    if cid not in pred_map:
        pred_map[cid] = row["Predicted_Label"]

print(f"  Unique claims in predictions: {len(pred_map)}")

# Build submission list
submission = []
missing_pred = []

for entry in kg_results:
    claim_id = entry["ID"]
    # Extract evidence text strings from the top_3_evidence list
    top3 = [ev["Evidence"] for ev in entry.get("top_3_evidence", [])]

    label = pred_map.get(claim_id)
    if label is None:
        missing_pred.append(claim_id)
        label = "NEI"  # fallback

    submission.append({
        "ID": claim_id,
        "top_3_evidence": top3,
        "Prediction": label
    })

print(f"  Total claims in submission: {len(submission)}")
if missing_pred:
    print(f"  WARNING: {len(missing_pred)} claims had no prediction (defaulted to NEI): {missing_pred[:5]}…")

print(f"Writing {OUTPUT} …")
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(submission, f, ensure_ascii=False, indent=2)

print("Done.")
print(f"\nSample entries:")
for entry in submission[:2]:
    print(json.dumps(entry, ensure_ascii=False, indent=2)[:400])
    print("---")
