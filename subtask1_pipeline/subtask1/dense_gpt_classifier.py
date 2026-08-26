import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from subtask1.evaluate import evaluate_retrieval, evaluate_veracity, format_submission_row, write_submission

SYSTEM_PROMPT = (
    "You are a fact-checking assistant. Given a CLAIM and one EVIDENCE passage, "
    "decide whether the evidence SUPPORTS or REFUTES the claim. "
    "Reply with exactly one word: SUPPORTS or REFUTES. No other text."
)


def load_gpt_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_gpt_cache(cache: dict, cache_path: Path):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def classify_one(client, model: str, claim_text: str, evidence_text: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=5,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CLAIM: {claim_text}\n\nEVIDENCE: {evidence_text}"},
        ],
    )
    answer = (resp.choices[0].message.content or "").strip().upper()
    return "REFUTES" if "REFUTE" in answer else "SUPPORTS"  # default to SUPPORTS on any garbled reply


def run_dense_gpt_pipeline(
    dense_rows: list,
    pool_id_to_text: dict,
    pool_text_to_id: dict,
    gold_evidence_by_id: dict,   # {} if the dataset has no gold at all (e.g. official test set)
    gold_label_by_id: dict,      # {} if the dataset has no gold at all
    cache_path: Path,
    results_dir: Path,
    dataset_label: str,
    max_workers: int = 10,
    output_prefix: str = "dense_gpt",     # filenames: {output_prefix}_submission.json / _detailed.json
    retrieval_label: str = "Dense-only",  # shown in the printed/summary headers -- name the retrieval that fed this
):
    import config
    from openai import OpenAI

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is empty -- set it in .env before running this.")
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    model = config.OPENAI_MODEL or "gpt-4o-mini"

    submission_path = results_dir / f"{output_prefix}_submission.json"
    detailed_path = results_dir / f"{output_prefix}_detailed.json"
    summary_path = results_dir / "summary.txt"

    # ---- Stage 1: top-3 (already computed by the caller -- Dense-only or
    # KG-boosted, whichever retrieval_label says -- no recompute here) ----
    top3_by_claim = {}
    retrieval_results = []
    for row in dense_rows:
        top3_ids = [e["ID"] for e in row["top10_evidence"][:3]]
        top3_by_claim[row["ID"]] = top3_ids
        gold_text = gold_evidence_by_id.get(row["ID"])
        if gold_text:
            retrieval_results.append({"claim_item": {"Evidence": gold_text}, "retrieved_ids": top3_ids})

    metrics = None
    if retrieval_results:
        metrics = evaluate_retrieval(retrieval_results, pool_text_to_id, k=3)
        print(f"\n=== Retrieval metrics ({retrieval_label}, {len(retrieval_results)}/{len(dense_rows)} claims with gold) ===")
        print(metrics)
    else:
        print("\nNo gold evidence available for this dataset -- skipping Success@3/nDCG@3.")

    # ---- Stage 2: GPT-4o-mini veracity classification (top-1 evidence, concurrent, cached) ----
    gpt_cache = load_gpt_cache(cache_path)
    to_classify = [row for row in dense_rows if row["ID"] not in gpt_cache]
    print(f"\n{len(dense_rows) - len(to_classify)}/{len(dense_rows)} already classified (cache hit). "
          f"{len(to_classify)} left to call GPT-4o-mini for.")

    def _work(row):
        top1_id = top3_by_claim[row["ID"]][0] if top3_by_claim[row["ID"]] else None
        top1_text = pool_id_to_text.get(top1_id, "") if top1_id else ""
        pred = classify_one(client, model, row["Claim"], top1_text or " ")
        return row["ID"], pred

    if to_classify:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_work, row): row for row in to_classify}
            done_count = 0
            for future in as_completed(futures):
                claim_id, pred = future.result()
                gpt_cache[claim_id] = pred
                done_count += 1
                if done_count % 100 == 0:
                    print(f"  {done_count}/{len(to_classify)} classified...")
                    save_gpt_cache(gpt_cache, cache_path)  # periodic save, don't lose progress on interruption
        save_gpt_cache(gpt_cache, cache_path)

    veracity_golds_and_preds = [
        (gold_label_by_id[row["ID"]], gpt_cache[row["ID"]])
        for row in dense_rows if row["ID"] in gold_label_by_id
    ]
    v_metrics = None
    if veracity_golds_and_preds:
        golds, preds = zip(*veracity_golds_and_preds)
        v_metrics = evaluate_veracity(list(preds), list(golds))
        print(f"\n=== Veracity metrics (GPT-4o-mini, {len(veracity_golds_and_preds)}/{len(dense_rows)} claims with gold label) ===")
        print(f"macro_F1: {v_metrics['macro_F1']:.4f}")
        print(f"Confusion matrix {v_metrics['confusion_matrix_labels']}: {v_metrics['confusion_matrix']}")
    else:
        print("\nNo gold veracity labels available for this dataset -- skipping macro F1.")

    # ---- Write official submission schema + detailed file ----
    output_rows = []
    detailed_rows = []
    for row in dense_rows:
        top3_ids = top3_by_claim[row["ID"]]
        pred = gpt_cache[row["ID"]]
        if len(top3_ids) == 3:
            output_rows.append(format_submission_row(row["ID"], top3_ids, pool_id_to_text, pred))
        detailed_rows.append({
            "ID": row["ID"],
            "Claim": row["Claim"],
            "top_3_evidence": [
                {"ID": e["ID"], "Evidence": e["Evidence"], "score": e["score"]}
                for e in row["top10_evidence"][:3]
            ],
            "Prediction": pred,
        })

    results_dir.mkdir(parents=True, exist_ok=True)
    write_submission(output_rows, str(submission_path),
                      expected_ids=[row["ID"] for row in dense_rows if len(top3_by_claim[row["ID"]]) == 3])

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_rows, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(detailed_rows)} detailed rows (claim + top-3 evidence + score + prediction) -> {detailed_path}")

    summary_lines = [f"=== {retrieval_label} retrieval + GPT-4o-mini veracity ({dataset_label}) ==="]
    if metrics:
        summary_lines.append(
            f"Retrieval -- n={metrics['n']}  Success@3={metrics['Success@3']:.4f}  nDCG@3={metrics['nDCG@3']:.4f}"
        )
    else:
        summary_lines.append("Retrieval -- no gold evidence available, Success@3/nDCG@3 not computed")
    if v_metrics:
        summary_lines.append(f"Veracity  -- n={len(veracity_golds_and_preds)}  macro_F1={v_metrics['macro_F1']:.4f}")
        summary_lines.append(f"Confusion matrix {v_metrics['confusion_matrix_labels']}: {v_metrics['confusion_matrix']}")
    else:
        summary_lines.append("Veracity  -- no gold labels available, macro F1 not computed")
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"Wrote summary -> {summary_path}")
