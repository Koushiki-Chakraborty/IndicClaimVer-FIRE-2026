"""
STEP 9: Fetch Full Article.

Single responsibility: given a URL, fetch the page and extract clean,
readable article text -- not raw HTML, not nav/ads/boilerplate, not
just the short snippet the search API already gave us. Verdicts should
be grounded in the actual article, not a 1-2 sentence snippet.

Uses trafilatura, a library purpose-built for exactly this job
(main-content extraction from arbitrary news/article HTML across many
languages, including Bengali/Hindi -- handles this far better than a
generic tag-stripping approach). Falls back to a lightweight
BeautifulSoup strip if trafilatura extracts nothing, and returns ""
(never raises) on total failure, so one dead/blocked/paywalled URL
never crashes a batch run over hundreds of claims -- verification.py
falls back to the evidence item's existing `snippet` field when this
returns empty.
"""
import requests

from . import config

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _fallback_extract(html: str) -> str:
    """Small BeautifulSoup fallback used only when trafilatura extracts
    nothing -- strips script/style/nav/header/footer/ads and joins
    remaining visible text. Lower precision than trafilatura (may
    include some boilerplate), but better than returning nothing."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    return " ".join(soup.stripped_strings).strip()


def fetch_and_extract(url: str, timeout: int = None) -> str:
    """Returns clean article text, truncated to config.MAX_ARTICLE_CHARS.
    Returns "" on ANY failure (network error, non-200 status, empty
    extraction) -- never raises, by design, so callers can always fall
    back to the snippet they already have."""
    timeout = timeout or config.ARTICLE_FETCH_TIMEOUT_SECONDS

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"[article_fetcher] WARNING: could not fetch {url}: {e}")
        return ""

    text = ""
    try:
        import trafilatura
        text = trafilatura.extract(html, favor_recall=True) or ""
    except Exception as e:
        print(f"[article_fetcher] WARNING: trafilatura failed for {url}: {e}")

    if not text.strip():
        try:
            text = _fallback_extract(html)
        except Exception as e:
            print(f"[article_fetcher] WARNING: fallback extraction failed for {url}: {e}")
            text = ""

    text = text.strip()
    if len(text) > config.MAX_ARTICLE_CHARS:
        text = text[: config.MAX_ARTICLE_CHARS]
    return text


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "https://en.wikipedia.org/wiki/Bengal"
    text = fetch_and_extract(url)
    print(f"Extracted {len(text)} character(s) from {url}:\n")
    print(text[:1000])