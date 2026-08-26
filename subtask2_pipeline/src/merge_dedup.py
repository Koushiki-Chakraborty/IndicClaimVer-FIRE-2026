"""
STEP 4: Merge + Deduplicate.

Combines results from multiple queries (native-language + English) into
one pool, removing duplicate sources by URL.

Why this matters: without dedup, an article that shows up in both the
native-language and English query results gets counted twice — which
silently inflates its score in every ranking step that comes after
(BM25, embeddings, entity-overlap), unfairly favoring duplicated
evidence over evidence that only appears once.
"""


def merge_and_dedup(*result_lists) -> list:
    """Takes any number of result lists (each a list of
    {"title","url","snippet"} dicts) and returns one combined,
    URL-deduplicated list. First occurrence of a URL wins."""
    seen_urls = set()
    merged = []
    for results in result_lists:
        for r in results:
            url = r.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            merged.append(r)
    return merged


if __name__ == "__main__":
    # Quick sanity test with fake data
    list_a = [
        {"title": "Article 1", "url": "http://example.com/a", "snippet": "..."},
        {"title": "Article 2", "url": "http://example.com/b", "snippet": "..."},
    ]
    list_b = [
        {"title": "Article 2 (dup)", "url": "http://example.com/b", "snippet": "..."},
        {"title": "Article 3", "url": "http://example.com/c", "snippet": "..."},
    ]
    merged = merge_and_dedup(list_a, list_b)
    print(f"Merged {len(list_a)} + {len(list_b)} -> {len(merged)} unique results")
    for r in merged:
        print(" -", r["title"])
