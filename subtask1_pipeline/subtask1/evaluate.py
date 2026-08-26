import json
import math

from sklearn.metrics import f1_score, confusion_matrix


def success_at_k(retrieved_ids: list, gold_id: str, k: int = 3) -> int:
    return int(gold_id in retrieved_ids[:k])


def ndcg_at_k(retrieved_ids: list, gold_id: str, k: int = 3) -> float:
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid == gold_id:
            return 1.0 / math.log2(i + 2)  # single relevant doc -> IDCG = 1
    return 0.0


def evaluate_retrieval(results: list, pool_text_to_id: dict, k: int = 3) -> dict:
    """results: list of {'claim_item': dict, 'retrieved_ids': list}"""
    s_at_k, ndcg = 0.0, 0.0
    n = len(results)
    for r in results:
        gold_id = pool_text_to_id[r["claim_item"]["Evidence"]]
        s_at_k += success_at_k(r["retrieved_ids"], gold_id, k)
        ndcg += ndcg_at_k(r["retrieved_ids"], gold_id, k)
    return {f"Success@{k}": s_at_k / n if n else 0.0,
            f"nDCG@{k}": ndcg / n if n else 0.0,
            "n": n}


def evaluate_veracity(preds: list, golds: list) -> dict:
    labels = ["REFUTES", "SUPPORTS"]
    f1 = f1_score(golds, preds, average="macro", labels=labels, zero_division=0)
    cm = confusion_matrix(golds, preds, labels=labels)
    return {
        "macro_F1": f1,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "n": len(golds),
    }


def evaluate_by_language(results: list, pool_text_to_id: dict,
                          lang_fn, k: int = 3) -> dict:
    """Breaks retrieval results down per-language -- this is the table
    that actually validates or disproves the KG-boost design, since the
    effect is expected to be uneven across languages/entity types, not
    a uniform lift."""
    by_lang = {}
    langs = {lang_fn(r["claim_item"]) for r in results}
    for lang in langs:
        subset = [r for r in results if lang_fn(r["claim_item"]) == lang]
        by_lang[lang] = evaluate_retrieval(subset, pool_text_to_id, k)
    return by_lang


# ---------------------------------------------------------------------
# Official Subtask 1 submission format
# ---------------------------------------------------------------------
VALID_LABELS = ("SUPPORTS", "REFUTES")


def format_submission_row(claim_id: str, top3_evidence_ids: list,
                           pool_id_to_text: dict, prediction: str) -> dict:
    """One row of the official schema:
    {"ID", "top_3_evidence": [text, text, text], "Prediction": "SUPPORTS"|"REFUTES"}.
    top_3_evidence is EVIDENCE TEXT (not IDs) per spec, ranked most-to-least
    relevant -- callers must pass ids already in that order.

    Raises rather than silently emitting an invalid row: the spec
    explicitly disallows empty evidence lists and any label other than
    SUPPORTS/REFUTES, and a submission-format bug is far cheaper to catch
    here than to discover from a rejected submission."""
    if not top3_evidence_ids:
        raise ValueError(f"{claim_id}: no evidence retrieved -- empty list is not permitted.")
    if len(top3_evidence_ids) != 3:
        raise ValueError(f"{claim_id}: need exactly 3 evidence items, got {len(top3_evidence_ids)}.")
    if prediction not in VALID_LABELS:
        raise ValueError(f"{claim_id}: prediction must be one of {VALID_LABELS}, got {prediction!r}.")
    try:
        evidence_texts = [pool_id_to_text[eid] for eid in top3_evidence_ids]
    except KeyError as e:
        raise ValueError(f"{claim_id}: retrieved evidence id {e} not found in the evidence pool.")
    return {"ID": claim_id, "top_3_evidence": evidence_texts, "Prediction": prediction}


def write_submission(rows: list, path: str, expected_ids: list = None):
    """Writes the submission JSON and validates it against the spec before
    ever touching disk with the final file -- catches a format bug at
    generation time, not at rejection time."""
    validate_submission(rows, expected_ids)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(rows)} predictions to {path} (schema validated)")


def validate_submission(rows, expected_ids: list = None):
    """Checks every hard requirement from the task's submission spec:
    valid rows, exactly-3 non-empty evidence texts, SUPPORTS/REFUTES-only
    predictions, unique IDs, and (if expected_ids is given) exact coverage
    of the test set with no extras and no omissions. Raises on the first
    violation found; returns True if everything checks out. `rows` can
    be a list (already loaded) or a path to a JSON file to load and check."""
    if isinstance(rows, str):
        with open(rows, encoding="utf-8") as f:
            rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("Submission must be a JSON array.")

    seen_ids = set()
    for row in rows:
        if not isinstance(row, dict) or set(row.keys()) != {"ID", "top_3_evidence", "Prediction"}:
            raise ValueError(f"Row has the wrong shape (expected exactly ID/top_3_evidence/Prediction): {row}")
        rid = row["ID"]
        if rid in seen_ids:
            raise ValueError(f"Duplicate ID in submission: {rid!r} -- each ID must appear exactly once.")
        seen_ids.add(rid)
        ev = row["top_3_evidence"]
        if not isinstance(ev, list) or len(ev) != 3:
            raise ValueError(f"{rid}: top_3_evidence must be a list of exactly 3 items, got {ev!r}.")
        if any(not isinstance(e, str) or not e.strip() for e in ev):
            raise ValueError(f"{rid}: top_3_evidence must contain non-empty evidence text, got {ev!r}.")
        if row["Prediction"] not in VALID_LABELS:
            raise ValueError(f"{rid}: Prediction must be one of {VALID_LABELS}, got {row['Prediction']!r}.")

    if expected_ids is not None:
        expected = set(expected_ids)
        missing = expected - seen_ids
        extra = seen_ids - expected
        if missing:
            raise ValueError(f"Missing {len(missing)} required claim ID(s) from submission: {sorted(missing)[:10]}...")
        if extra:
            raise ValueError(f"Submission has {len(extra)} ID(s) not in the test set: {sorted(extra)[:10]}...")
    return True


if __name__ == "__main__":
    # sanity check
    retrieved = ["EV/1", "EV/2", "EV/3"]
    print("Testing success_at_k / ndcg_at_k:")
    print("Gold at rank 1:", "Success@3 =", success_at_k(retrieved, "EV/1", 3), "nDCG@3 =", ndcg_at_k(retrieved, "EV/1", 3))
    print("Success@3 (hit at rank 2):", success_at_k(retrieved, "EV/2", 3))
    print("nDCG@3 (hit at rank 2):", ndcg_at_k(retrieved, "EV/2", 3))
    print("Success@3 (miss):", success_at_k(retrieved, "EV/9", 3))

    print("\nTesting evaluate_veracity:")
    fake_preds = ["SUPPORTS", "REFUTES", "SUPPORTS", "SUPPORTS", "REFUTES"]
    fake_golds = ["SUPPORTS", "REFUTES", "REFUTES", "SUPPORTS", "REFUTES"]
    veracity_result = evaluate_veracity(fake_preds, fake_golds)
    print("macro_F1:", veracity_result["macro_F1"])
    print("confusion_matrix:", veracity_result["confusion_matrix"])
