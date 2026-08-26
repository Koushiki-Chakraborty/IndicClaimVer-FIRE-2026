import json, sys, time
from pathlib import Path
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from subtask1 import data_loader
from subtask1.evaluate import ndcg_at_k, success_at_k
from subtask1.embeddings import (
    dense_retrieve_partitioned, dense_retrieve_vec,
    encode_batch, prepare_pool_embeddings,
)


def tokenize(text):
    return text.lower().split()


def build_bm25_indexes(pool):
    by_lang = data_loader.build_pool_by_lang(pool)
    indexes = {}
    for lang, items in by_lang.items():
        if lang not in {"hi", "bn", "en"}:
            continue
        corpus = [tokenize(item["Evidence"]) for item in items]
        indexes[lang] = BM25Okapi(corpus)
    return by_lang, indexes


def evaluate_gate2(dev, pool):
    pool_text_to_id = data_loader.build_pool_text_to_id(pool)
    pool_by_lang, bm25_indexes = build_bm25_indexes(pool)
    prepare_pool_embeddings(pool)
    candidate_id_sets = {
        lang: {item["ID"] for item in items}
        for lang, items in pool_by_lang.items()
    }
    dev_texts = [c["Text"] for c in dev]
    dev_vecs = {}
    if dev_texts:
        vectors = encode_batch(dev_texts, batch_size=64, save_every=200)
        for t, v in zip(dev_texts, vectors):
            dev_vecs[t] = v

    results, misrouted = [], []
    for claim in dev:
        claim_text = claim["Text"]
        claim_lang = data_loader.detect_lang(claim_text)
        bucket = "code-mixed" if data_loader.is_code_mixed(claim_text) else claim_lang
        gold_id = pool_text_to_id[claim["Evidence"]]
        candidate_items = pool_by_lang.get(claim_lang, [])
        candidate_ids = [item["ID"] for item in candidate_items]

        if claim_lang in bm25_indexes and candidate_ids:
            scores = bm25_indexes[claim_lang].get_scores(tokenize(claim_text))
            ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            bm25_ids = [candidate_ids[i] for i in ranked_idx[:10]]
        else:
            bm25_ids = []

        claim_vec = dev_vecs.get(claim_text)
        dense_full_ids = dense_retrieve_vec(claim_vec, top_k=10) if claim_vec is not None else []
        dense_part_ids = (
            dense_retrieve_partitioned(claim_text, claim_vec, top_k=10)
            if claim_vec is not None else []
        )

        results.append({
            "claim_id": claim.get("ID"),
            "gold_id": gold_id,
            "bm25_retrieved_ids": bm25_ids,
            "dense_retrieved_ids": dense_full_ids,
            "dense_partitioned_ids": dense_part_ids,
            "lang": claim_lang,
            "bucket": bucket,
        })
        if gold_id not in candidate_id_sets.get(claim_lang, set()):
            misrouted.append(claim.get("ID"))
    return results, misrouted


def summarize_results(results, misrouted, dev_size):
    overall = list(results)
    buckets = {"hi": [], "bn": [], "en": [], "code-mixed": []}
    for r in results:
        buckets[r["bucket"]].append(r)

    def metrics(items, key):
        if not items:
            return 0.0, 0.0
        s = sum(success_at_k(r[key], r["gold_id"], 3) for r in items) / len(items)
        n = sum(ndcg_at_k(r[key], r["gold_id"], 3) for r in items) / len(items)
        return s, n

    labels = {"hi": "Hindi", "bn": "Bengali", "en": "English", "code-mixed": "Code-mixed"}
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log("")
    log("=== BM25 vs Dense full-pool vs Dense partitioned [dev n=1134] ===")
    cols = ["Language", "BM25 S@3", "BM25 nDCG", "Full S@3", "Full nDCG", "Part S@3", "Part nDCG", "n"]
    widths = [12, 10, 10, 10, 10, 10, 10, 4]
    hdr = "  ".join(f"{c:{w}}" for c, w in zip(cols, widths))
    log(hdr)
    log("-" * len(hdr))
    for name in ["hi", "bn", "en", "code-mixed"]:
        subset = buckets[name]
        if subset:
            bs, bn2 = metrics(subset, "bm25_retrieved_ids")
            fs, fn = metrics(subset, "dense_retrieved_ids")
            ps, pn = metrics(subset, "dense_partitioned_ids")
            vals = [labels[name], f"{bs:.3f}", f"{bn2:.3f}", f"{fs:.3f}", f"{fn:.3f}", f"{ps:.3f}", f"{pn:.3f}", str(len(subset))]
        else:
            vals = [labels[name], "0.000", "0.000", "0.000", "0.000", "0.000", "0.000", "0"]
        log("  ".join(f"{v:{w}}" for v, w in zip(vals, widths)))

    bs_all, bn_all = metrics(overall, "bm25_retrieved_ids")
    fs_all, fn_all = metrics(overall, "dense_retrieved_ids")
    ps_all, pn_all = metrics(overall, "dense_partitioned_ids")
    vals = ["Overall", f"{bs_all:.3f}", f"{bn_all:.3f}", f"{fs_all:.3f}", f"{fn_all:.3f}", f"{ps_all:.3f}", f"{pn_all:.3f}", str(len(overall))]
    log("  ".join(f"{v:{w}}" for v, w in zip(vals, widths)))
    log(f"\nMisrouted by language detection: {len(misrouted)} / {dev_size}")
    if misrouted:
        log("Examples: " + ", ".join(misrouted[:5]))
    return lines


def _method_summary_lines(method_label, results_key, results):
    """Build a short per-method summary table (Success@3 / nDCG@3 per language)."""
    buckets = {"hi": [], "bn": [], "en": [], "code-mixed": []}
    for r in results:
        buckets[r["bucket"]].append(r)

    def metrics(items):
        if not items:
            return 0.0, 0.0
        s = sum(success_at_k(r[results_key], r["gold_id"], 3) for r in items) / len(items)
        n = sum(ndcg_at_k(r[results_key], r["gold_id"], 3) for r in items) / len(items)
        return s, n

    labels = {"hi": "Hindi", "bn": "Bengali", "en": "English", "code-mixed": "Code-mixed"}
    cols = ["Language", "Success@3", "nDCG@3", "n"]
    widths = [12, 10, 10, 4]
    hdr = "  ".join(f"{c:{w}}" for c, w in zip(cols, widths))
    sep = "-" * len(hdr)

    lines = ["", f"=== {method_label} per-language metrics ===", hdr, sep]
    for name in ["hi", "bn", "en", "code-mixed"]:
        subset = buckets[name]
        s, n = metrics(subset)
        cnt = len(subset)
        row = [labels[name], f"{s:.3f}", f"{n:.3f}", str(cnt)]
        lines.append("  ".join(f"{v:{w}}" for v, w in zip(row, widths)))

    s_all, n_all = metrics(results)
    row_all = ["Overall", f"{s_all:.3f}", f"{n_all:.3f}", str(len(results))]
    lines.append("  ".join(f"{v:{w}}" for v, w in zip(row_all, widths)))
    return lines


def _write_method(root, method_dir, method_label, results_key, results):
    out = root / "results" / "devResult" / method_dir
    out.mkdir(parents=True, exist_ok=True)
    per_claim = [
        {"claim_id": r["claim_id"], "gold_id": r["gold_id"],
         "retrieved_ids": r[results_key], "lang": r["lang"], "bucket": r["bucket"]}
        for r in results
    ]
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(per_claim, f, indent=2, ensure_ascii=False)
    method_lines = _method_summary_lines(method_label, results_key, results)
    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(method_lines) + "\n")


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    train, dev, pool = data_loader.load_all()
    if not (train and pool):
        print("Need train_subtask1.json and evidence_pool.json. Aborting.")
        sys.exit(1)

    t0 = time.perf_counter()
    results, misrouted = evaluate_gate2(dev, pool)
    elapsed = time.perf_counter() - t0
    summary_lines = summarize_results(results, misrouted, len(dev))

    _write_method(root_dir, "bm25", "BM25", "bm25_retrieved_ids", results)
    _write_method(root_dir, "dense_full_pool", "Dense full-pool", "dense_retrieved_ids", results)
    _write_method(root_dir, "dense_partitioned", "Dense partitioned", "dense_partitioned_ids", results)

    output_dir = root_dir / "results" / "devResult"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "gate2_bm25_vs_dense_dev_results.json"
    summary_path = output_dir / "summary.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nElapsed: {elapsed:.2f}s")
    print(f"Legacy JSON: {json_path}")
    print("Per-method: results/devResult/bm25/, results/devResult/dense_full_pool/, results/devResult/dense_partitioned/")
