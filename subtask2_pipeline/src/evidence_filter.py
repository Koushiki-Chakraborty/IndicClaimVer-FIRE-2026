"""
OPTIONAL ADD-ON: Lightweight Evidence Filtering, run between
merge_dedup.py and bm25.py.

Purpose: drop obvious non-article noise from the merged retrieval pool
BEFORE it competes for BM25 / embedding / entity-graph ranking. This
pipeline's own logs show the failure mode this targets directly --
article_fetcher.py repeatedly fails to extract usable text from ToS
pages, cookie-consent pages, login walls, and social-media posts (e.g.
Facebook post URLs returning 400, TripAdvisor listings returning 403).
Those pages can still score competitively on keyword overlap alone and
occupy a top-3 fusion slot without ever contributing real evidence at
verification time. Removing them earlier means every ranking signal
downstream (BM25, embedding, entity overlap) scores over a cleaner
pool, and the final top-3 is more likely to be genuine article content.

DESIGN PRINCIPLES (all required by the "do not break the pipeline" brief):

1. STRICT SUBSET. filter_evidence() only ever REMOVES items; it never
   adds, reorders beyond removal, or transforms a surviving item's
   fields. Input and output are both the exact {"title","url","snippet"}
   shape merge_dedup.py already produces -- nothing downstream needs to
   change to consume this.

2. CONSERVATIVE. Rules only catch clear, low-precision-risk noise
   patterns (near-empty snippets, cookie/ToS/login/privacy-policy
   boilerplate, generic placeholder titles). A short-but-real news
   snippet is never targeted by these rules -- when in doubt, an item
   is KEPT, not dropped, because a false negative (junk that slips
   through) only costs what today's pipeline already tolerates, while a
   false positive (a real article dropped) would actively hurt recall.

3. NEVER STARVES DOWNSTREAM. If filtering would remove every single
   item (e.g. a genuinely bad retrieval where everything happens to
   match a noise pattern), this module logs a warning and returns the
   ORIGINAL, UNFILTERED pool instead -- so BM25/verification are never
   handed an empty list because of this add-on. Same graceful-fallback
   convention already used throughout this codebase (NER fallback in
   entity_extraction.py, translation fallback in query_generation.py,
   QID-linking fallback in entity_linking.py).

4. NO NEW DEPENDENCY. Pure string/regex checks over fields already
   present on every evidence item -- no network calls, no model
   loading, negligible runtime cost even over hundreds of claims.
"""
import re

# Minimum snippet length (characters) to be considered non-trivial.
# Chosen conservatively -- real news snippets from Tavily are typically
# well over 100 characters; this only catches near-empty/placeholder
# snippets, not genuinely short-but-real ones.
_MIN_SNIPPET_CHARS = 20

# URL path/query patterns strongly associated with non-article pages
# (cookie/consent/ToS/privacy/login/signup walls) rather than news
# content. Matched case-insensitively against the URL only -- never
# against title/snippet, to avoid accidentally dropping a real article
# that merely mentions e.g. "privacy" in its headline.
_NOISE_URL_PATTERNS = (
    r"/cookie-?policy",
    r"/cookies?/?$",
    r"/terms-?of-?service",
    r"/terms-?and-?conditions",
    r"/privacy-?policy",
    r"/login",
    r"/signin",
    r"/sign-?up",
    r"/subscribe",
    r"/account/",
)
_NOISE_URL_RE = re.compile("|".join(_NOISE_URL_PATTERNS), re.IGNORECASE)

# Title patterns that are near-universally boilerplate, not article
# content, when they make up the ENTIRE title (not just a substring of
# a longer real headline).
_NOISE_TITLE_PATTERNS = (
    r"^cookie policy$",
    r"^terms of service$",
    r"^terms and conditions$",
    r"^privacy policy$",
    r"^sign in$",
    r"^log in$",
    r"^login$",
    r"^page not found$",
    r"^404 not found$",
    r"^just a moment\.\.\.$",  # common Cloudflare interstitial title
)
_NOISE_TITLE_RE = re.compile(
    "|".join(_NOISE_TITLE_PATTERNS), re.IGNORECASE
)

# OPTIONAL ADD-ON (2026-07): social-media domains that showed up
# repeatedly in dev-set false-REFUTES errors as near-content-free
# "evidence" -- e.g. "Source 1 is a Facebook page with no relevant
# details" -- crowding out a top-3 verification slot that a real news
# article could have occupied instead. Deliberately NOT a blanket domain
# ban: a social-media URL with a substantive snippet (e.g. an official
# press release actually quoted in the snippet) is still kept, per the
# same "when in doubt, keep it" principle as the rest of this module.
# Only dropped when BOTH the domain matches AND the snippet is thin.
_SOCIAL_MEDIA_DOMAINS = (
    "facebook.com", "instagram.com", "twitter.com", "x.com",
    "tiktok.com", "pinterest.com", "t.me", "youtube.com",
)
_SOCIAL_MEDIA_MIN_SNIPPET_CHARS = 80  # higher bar than the general _MIN_SNIPPET_CHARS,
                                       # since these platforms are disproportionately
                                       # link/placeholder-only in practice (see above)


def _is_thin_social_media(item: dict) -> bool:
    url = (item.get("url") or "").strip().lower()
    snippet = (item.get("snippet") or "").strip()
    if not url:
        return False
    if not any(domain in url for domain in _SOCIAL_MEDIA_DOMAINS):
        return False
    return len(snippet) < _SOCIAL_MEDIA_MIN_SNIPPET_CHARS


def _is_noise(item: dict) -> bool:
    """True if this single evidence item looks like non-article noise
    rather than genuine retrieved content. Conservative by design --
    see module docstring."""
    title = (item.get("title") or "").strip()
    snippet = (item.get("snippet") or "").strip()
    url = (item.get("url") or "").strip()

    # Rule 1: effectively empty content -- no snippet worth ranking on,
    # and no title either (an item with a real title but thin snippet
    # is NOT dropped by this rule alone, to stay conservative).
    if len(snippet) < _MIN_SNIPPET_CHARS and not title:
        return True

    # Rule 2: URL matches a known non-article page pattern.
    if url and _NOISE_URL_RE.search(url):
        return True

    # Rule 3: title IS (not merely contains) a known boilerplate phrase.
    if title and _NOISE_TITLE_RE.match(title):
        return True

    # Rule 4 (OPTIONAL ADD-ON): thin/placeholder social-media content --
    # see _is_thin_social_media() docstring above.
    if _is_thin_social_media(item):
        return True

    return False


def filter_evidence(evidence_pool: list) -> list:
    """Takes the merged evidence pool (output of
    merge_dedup.merge_and_dedup()) and returns a new list with obvious
    non-article noise removed. STRICT SUBSET of the input -- order of
    surviving items is preserved, no fields are added or changed.

    Falls back to returning the ORIGINAL unfiltered pool if filtering
    would remove everything, so this stage can never starve downstream
    ranking/verification of candidates (see module docstring, point 3).
    """
    if not evidence_pool:
        return evidence_pool

    kept = [item for item in evidence_pool if not _is_noise(item)]

    if not kept:
        print(f"[evidence_filter] WARNING: filtering would have removed all "
              f"{len(evidence_pool)} item(s); falling back to the unfiltered "
              f"pool instead of returning an empty list.")
        return evidence_pool

    removed = len(evidence_pool) - len(kept)
    if removed:
        print(f"[evidence_filter] Removed {removed} noise item(s) "
              f"({len(evidence_pool)} -> {len(kept)}).")

    return kept


if __name__ == "__main__":
    # Smoke test with a mix of real-looking and noise items.
    fake_pool = [
        {
            "title": "Rajasthan HC seeks explanation from judicial magistrate",
            "url": "https://www.livelaw.in/rajasthan-hc-seeks-explanation",
            "snippet": "The Rajasthan High Court on Tuesday sought an explanation "
                       "from a judicial magistrate for refusing to grant bail...",
        },
        {
            "title": "Cookie Policy",
            "url": "https://example.com/cookie-policy",
            "snippet": "We use cookies to improve your experience.",
        },
        {
            "title": "Terms of service - Some Company",
            "url": "https://example.com/pages/terms-of-service",
            "snippet": "By using this site you agree to...",
        },
        {
            "title": "",
            "url": "https://example.com/empty",
            "snippet": "",
        },
        {
            "title": "Sign In",
            "url": "https://example.com/login",
            "snippet": "Please sign in to continue.",
        },
        {
            "title": "Short but real headline",
            "url": "https://news.example.com/article-123",
            "snippet": "Brief but genuine.",  # short, but has a real title -> kept
        },
    ]

    result = filter_evidence(fake_pool)
    print(f"\nKept {len(result)}/{len(fake_pool)} item(s):")
    for r in result:
        print(f"  - {r['title']!r}  ({r['url']})")