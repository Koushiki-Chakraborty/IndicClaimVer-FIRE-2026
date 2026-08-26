"""
Standalone MuRIL SUPPORTS/REFUTES classifier -- inference only.

This is the SAME fine-tuned checkpoint as subtask1_pipeline's
subtask1/muril_classifier.py (verified: notebook/FIRE2026_FINAL.ipynb,
notebook/FIRE_2026_HPT_01.ipynb, and classifier/config.json in this repo
are byte-for-byte identical to subtask1_pipeline's copies). It was
fine-tuned on Subtask 1's train/dev claim-evidence pairs, then reused
here as-is for Subtask 2 -- it was never trained on any Subtask 2 data.
There is no training code in this repo; the checkpoint was uploaded to
HuggingFace and is loaded directly from there.

This is a separate, standalone prediction path from run_pipeline.py's
Gemini-based verification.py -- it does not touch or depend on any part
of that pipeline beyond reading the same fusion_ranked.json files it
already produces.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config


class VeracityPredictor:
    """Loads config.MURIL_CLASSIFIER_MODEL (default
    nirnit-13/muril-supports-refutes) straight from the Hub. Label
    mapping is read from the checkpoint's own config.id2label, never
    hardcoded here."""

    def __init__(self, model_name: str = None, max_length: int = 192, device: str = None):
        self.model_name = model_name or config.MURIL_CLASSIFIER_MODEL
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[muril_classifier] Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        print(f"[muril_classifier] Loaded. Label map: {self.id2label}")

    @torch.no_grad()
    def predict_batch(self, claim_texts: list, evidence_texts: list, batch_size: int = 32) -> list:
        """claim_texts[i] paired with evidence_texts[i] -- same convention
        as the training notebook's tokenize_fn (claim first, evidence
        second). Returns a list of (label, confidence) tuples, same
        length and order as the inputs."""
        assert len(claim_texts) == len(evidence_texts), \
            "claim_texts and evidence_texts must be the same length (paired 1:1)"
        preds = []
        for start in range(0, len(claim_texts), batch_size):
            batch_claims = claim_texts[start:start + batch_size]
            batch_evidence = evidence_texts[start:start + batch_size]
            enc = self.tokenizer(
                batch_claims, batch_evidence,
                truncation=True, padding=True, max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            batch_pred_ids = torch.argmax(probs, dim=-1).tolist()
            batch_confidences = probs.max(dim=-1).values.tolist()
            preds.extend(
                (self.id2label[i], conf) for i, conf in zip(batch_pred_ids, batch_confidences)
            )
        return preds

    def predict_one(self, claim_text: str, evidence_text: str) -> tuple:
        return self.predict_batch([claim_text], [evidence_text])[0]


if __name__ == "__main__":
    predictor = VeracityPredictor()
    label, confidence = predictor.predict_one(
        "The Prime Minister visited the flood-affected region on Tuesday.",
        "The Prime Minister's office confirmed a visit to the flood-hit "
        "district on Tuesday to review relief operations.",
    )
    print(f"Prediction: {label} (confidence={confidence:.4f})")
