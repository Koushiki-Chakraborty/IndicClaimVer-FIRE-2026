import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from subtask1 import data_loader
from subtask1.evaluate import evaluate_retrieval

CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "bge_m3"
POOL_EMBED_CACHE = CACHE_DIR / "pool_embeddings.pkl"
CLAIM_EMBED_CACHE = CACHE_DIR / "claim_embeddings.pkl"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "devResult" / "bge_m3"

BUCKET_LABELS = {"hi": "Hindi", "bn": "Bengali", "en": "English", "code-mixed": "Code-mixed"}


def _load_or_encode(texts_by_id: dict, cache_path: Path, model, batch_size=32) -> dict:
    cache = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    missing_ids = [i for i in texts_by_id if i not in cache]
    if missing_ids:
        print(f"Encoding {len(missing_ids)}/{len(texts_by_id)} new texts (BGE-M3)...")
        missing_texts = [texts_by_id[i] for i in missing_ids]
        embeddings = model.encode(missing_texts, batch_size=batch_size, show_progress_bar=True,
                                    normalize_embeddings=True)
        for i, emb in zip(missing_ids, embeddings):
            cache[i] = emb
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"Cached ({len(cache)} total) -> {cache_path}")
    else:
        print(f"All {len(texts_by_id)} texts already cached -> {cache_path}")
    return cache


def main(limit: int = None):
    from sentence_transformers import SentenceTransformer

    train, dev, pool = data_loader.load_all()
    all_claims = train + dev
    if limit:
        all_claims = all_claims[:limit]
    print(f"Loaded {len(all_claims)} claims, {len(pool)} evidence.")

    print("Loading BAAI/bge-m3 (downloads the model the first time, ~2.2GB)...")
    model = SentenceTransformer("BAAI/bge-m3")

    pool_texts_by_id = {p["ID"]: p["Evidence"] for p in pool}
    claim_texts_by_id = {c["ID"]: c["Text"] for c in all_claims}

    pool_embeddings = _load_or_encode(pool_texts_by_id, POOL_EMBED_CACHE, model)
    claim_embeddings = _load_or_encode(claim_texts_by_id, CLAIM_EMBED_CACHE, model)

    pool_by_lang = data_loader.build_pool_by_lang(pool)
    lang_matrix = {}
    lang_ids = {}
    for lang, items in pool_by_lang.items():
        ids = [item["ID"] for item in items]
        lang_ids[lang] = ids
        lang_matrix[lang] = np.array([pool_embeddings[i] for i in ids])

    gold_evidence_by_id = {c["ID"]: c["Evidence"] for c in all_claims if "Evidence" in c}
    pool_text_to_id = data_loader.build_pool_text_to_id(pool)

    output_rows = []
    retrieval_by_bucket = {"hi": [], "bn": [], "en": [], "code-mixed": []}

    for c in all_claims:
        claim_text = c["Text"]
        lang = data_loader.detect_lang(claim_text)
        query_vec = claim_embeddings[c["ID"]]
        ids = lang_ids.get(lang, [])
        matrix = lang_matrix.get(lang)
        if matrix is None or len(ids) == 0:
            continue
        sims = matrix @ query_vec  # already normalized during encode() -> this is cosine similarity
        top_idx = np.argsort(-sims)[:3]
        top3 = [(ids[i], float(sims[i])) for i in top_idx]

        output_rows.append({
            "ID": c["ID"],
            "Claim": claim_text,
            "top_3_evidence": [
                {"ID": eid, "Evidence": pool_texts_by_id.get(eid, ""), "score": score} for eid, score in top3
            ],
        })

        gold_text = gold_evidence_by_id.get(c["ID"])
        if gold_text:
            bucket = "code-mixed" if data_loader.is_code_mixed(claim_text) else lang
            retrieval_by_bucket.setdefault(bucket, [])
            retrieval_by_bucket[bucket].append(
                {"claim_item": {"Evidence": gold_text}, "retrieved_ids": [eid for eid, _ in top3]}
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(output_rows)} rows -> {out_json}")

    all_results = [r for bucket_results in retrieval_by_bucket.values() for r in bucket_results]
    overall = evaluate_retrieval(all_results, pool_text_to_id, k=3) if all_results else None

    summary_lines = ["=== BGE-M3 Dense retrieval (data/ train+dev) ==="]
    if overall:
        summary_lines.append(f"{'Language':<12} {'Success@3':<12} {'nDCG@3':<12} {'n':<8}")
        summary_lines.append("-" * 44)
        for key in ["hi", "bn", "en", "code-mixed"]:
            if retrieval_by_bucket[key]:
                m = evaluate_retrieval(retrieval_by_bucket[key], pool_text_to_id, k=3)
                summary_lines.append(f"{BUCKET_LABELS[key]:<12} {m['Success@3']:<12.3f} {m['nDCG@3']:<12.3f} {m['n']:<8}")
        summary_lines.append(f"{'Overall':<12} {overall['Success@3']:<12.3f} {overall['nDCG@3']:<12.3f} {overall['n']:<8}")
    else:
        summary_lines.append("No gold evidence available in this slice.")
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    out_summary = RESULTS_DIR / "summary.txt"
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\nWrote summary -> {out_summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N claims (for a quick test)")
    args = parser.parse_args()
    main(limit=args.limit)
