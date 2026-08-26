"""
STEP 5: BM25 Ranking.

Single responsibility: given ONE query string and an evidence pool,
tokenize, score with BM25, normalize the scores, rank, and return
exactly: title, url, snippet, bm25_score_raw, bm25_score_norm,
bm25_rank.

No script detection, no language routing, no "matched_script", no
embeddings, no NER, no entity overlap, no Knowledge Graph, no
credibility weighting, no fusion. This module does not know or care
what language the query or documents are in -- deciding what single
query string to pass in (e.g. combining a native-language query and an
English query into one string so a single ranking pass covers evidence
in either language) is the CALLER's responsibility, not this module's.

Uses a proper BM25 corpus (all documents in evidence_pool indexed
together), scored against one query in a single pass -- this is
standard rank_bm25 usage and gives meaningful IDF weighting across the
real evidence pool, rather than scoring each document in isolation.
"""
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list:
    return text.lower().split()


def _minmax_normalize(scores: list) -> list:
    """Scales raw BM25 scores to 0-1. If all scores are equal (including
    all-zero), returns all 0.5 to avoid divide-by-zero / false signal."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def rank_evidence(query: str, evidence_pool: list) -> list:
    """query: a single search query string (callers decide its content
    -- e.g. combined native + English text -- this function just scores
    against whatever string it's given).

    evidence_pool: list of {"title", "url", "snippet"} dicts (the
    output of merge_dedup.merge_and_dedup).

    Returns a NEW list, sorted by BM25 score descending, each item
    containing exactly: title, url, snippet, bm25_score_raw,
    bm25_score_norm, bm25_rank. No other field is added.
    """
    if not evidence_pool:
        return []

    query_tokens = _tokenize(query)
    doc_texts = [f"{r.get('title', '')} {r.get('snippet', '')}" for r in evidence_pool]
    tokenized_corpus = [_tokenize(t) for t in doc_texts]

    bm25 = BM25Okapi(tokenized_corpus)
    raw_scores = [float(s) for s in bm25.get_scores(query_tokens)]
    norm_scores = _minmax_normalize(raw_scores)

    combined = sorted(
        zip(evidence_pool, raw_scores, norm_scores),
        key=lambda item: -item[1],
    )

    ranked = []
    for rank, (r, raw, norm) in enumerate(combined, start=1):
        ranked.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("snippet", ""),
            "bm25_score_raw": raw,
            "bm25_score_norm": norm,
            "bm25_rank": rank,
        })
    return ranked


if __name__ == "__main__":
    # Smoke test: combined bilingual query, real Bamangachi evidence pool
    combined_query = (
        "Bamangachi Station lockdown liquor parties crime "
        "লকডাউন বামনগাছি স্টেশন মদের আসর দুষ্কৃতী"
    )

    fake_pool = [
        {
            "title": "লকডাউনে শুনশান বামনগাছি স্টেশনই দুষ্কৃতীদের স্বর্গরাজ্য",
            "url": "https://www.sangbadpratidin.in/bengal/some-youth-drinking-alcohol-in-bamangachi-station-during-lockdown/pid/392955",
            "snippet": "এই লকডাউনে দুষ্কৃতীদের স্বর্গরাজ্যে পরিণত হয়েছে বামনগাছি স্টেশন।",
        },
        {
            "title": "Terms of service - Lockdown Liquor & Co",
            "url": "http://lockdown-liquor.com/pages/private-corporate-events",
            "snippet": "By clicking 'Yes', you confirm that you are over 18 years old...",
        },
        {
            "title": "Police arrest two liquor smugglers during lockdown in Rajasthan",
            "url": "https://www.ficcicascade.in/police-arrest-two-liquor-smugglers-during-lockdown-amid-lockdown-in-rajasthan",
            "snippet": "Ganganagar: To prevent the spread of coronavirus infection...",
        },
    ]

    ranked = rank_evidence(combined_query, fake_pool)
    print("Ranked results (best first):\n")
    for r in ranked:
        print(f"#{r['bm25_rank']}  raw={r['bm25_score_raw']:.3f}  norm={r['bm25_score_norm']:.2f}")
        print(f"    {r['title']}")
        print()