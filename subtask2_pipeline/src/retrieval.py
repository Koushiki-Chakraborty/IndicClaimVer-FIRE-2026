"""
STEP 3: Text Retrieval.

Sends a query to a web search API and returns structured results.
Uses Tavily by default (free tier available at tavily.com). To swap in
a different provider (Serper, Bing), only this file needs to change —
every other module just consumes the {"title","url","snippet"} shape.
"""
from . import config

_client = None


def _get_client():
    global _client
    if _client is None:
        from tavily import TavilyClient
        if not config.TAVILY_API_KEY:
            raise RuntimeError(
                "TAVILY_API_KEY not set. Add it to your .env file or "
                "export it as an environment variable."
            )
        _client = TavilyClient(api_key=config.TAVILY_API_KEY)
    return _client


def search(query: str, top_k: int = None) -> list:
    """Returns list of {"title": str, "url": str, "snippet": str}.

    OPTIONAL ADD-ON (2026-07): passes topic="news" to Tavily. Diagnosed
    directly from dev-set F1 errors: several false-REFUTES cases were
    caused by retrieval surfacing generic/reference content instead of
    the actual news event -- e.g. a claim mentioning "LS polls" (Lok
    Sabha) pulled back Unix `ls` command documentation, and a claim
    about a new parliamentary panel pulled back a generic explainer on
    how the CJI is normally appointed. Tavily's "news" topic mode biases
    results toward actual news articles rather than general web/reference
    pages, directly targeting this failure pattern. Falls back to the
    default (unscoped) search automatically if this parameter causes any
    error, so this can never make retrieval worse than before -- it can
    only narrow toward more relevant results or no-op on failure.
    """
    top_k = top_k or config.TOP_K_PER_QUERY
    client = _get_client()

    try:
        resp = client.search(query=query, max_results=top_k, topic="news")
    except Exception as e:
        print(f"[retrieval] WARNING: news-topic search failed for query "
              f"'{query}' ({e}); retrying without topic scoping.")
        try:
            resp = client.search(query=query, max_results=top_k)
        except Exception as e2:
            print(f"[retrieval] WARNING: search failed for query '{query}': {e2}")
            return []

    results = []
    for r in resp.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        })
    return results


if __name__ == "__main__":
    results = search("Bamangachi station lockdown crime liquor")
    for r in results:
        print(f"- {r['title']}  ({r['url']})")
        print(f"  {r['snippet'][:150]}...")
        print()