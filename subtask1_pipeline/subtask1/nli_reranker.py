import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config

_tokenizer = None
_model = None
_entailment_idx = None


def _load_model():
    global _tokenizer, _model, _entailment_idx
    if _model is not None:
        return
    print(f"[nli_reranker] Loading {config.NLI_MODEL} (pretrained entailment head, not fine-tuned)...")
    _tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL)
    _model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL)
    _model.eval()
    id2label = {k: v.lower() for k, v in _model.config.id2label.items()}
    _entailment_idx = next(i for i, label in id2label.items() if "entail" in label)
    print(f"[nli_reranker] Loaded. Label map: {id2label} -> entailment index = {_entailment_idx}")


def rerank_top10(dense_rows: list, batch_size: int = 16) -> list:
    """dense_rows: same shape as dense_top10.json (list of {ID, Claim,
    top10_evidence: [{ID, Evidence, score}, ...]}). Returns a NEW list,
    same shape, with each row's top10_evidence reordered by entailment
    probability (evidence=premise, claim=hypothesis) instead of Dense
    cosine similarity -- "score" is replaced by that entailment
    probability so downstream code (which just reads "score") needs no
    changes to consume either ranking."""
    _load_model()

    pairs = []  # (row_idx, premise_text, hypothesis_text)
    for ri, row in enumerate(dense_rows):
        for cand in row["top10_evidence"]:
            pairs.append((ri, cand["Evidence"], row["Claim"]))

    scores = [None] * len(pairs)
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            premises = [p[1] for p in batch]
            hypotheses = [p[2] for p in batch]
            enc = _tokenizer(premises, hypotheses, truncation=True, padding=True,
                              max_length=256, return_tensors="pt")
            logits = _model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, _entailment_idx].tolist()
            for offset, prob in enumerate(probs):
                scores[start + offset] = prob
            batch_num = start // batch_size
            if batch_num % 50 == 0:
                print(f"[nli_reranker] {start}/{len(pairs)} pairs scored...")

    by_row = [[] for _ in dense_rows]
    pair_idx = 0
    for ri, row in enumerate(dense_rows):
        for cand in row["top10_evidence"]:
            cand_with_score = {"ID": cand["ID"], "Evidence": cand["Evidence"], "score": scores[pair_idx]}
            by_row[ri].append(cand_with_score)
            pair_idx += 1

    reranked_rows = []
    for ri, row in enumerate(dense_rows):
        reranked = sorted(by_row[ri], key=lambda c: c["score"], reverse=True)
        reranked_rows.append({"ID": row["ID"], "Claim": row["Claim"], "top10_evidence": reranked})

    return reranked_rows
