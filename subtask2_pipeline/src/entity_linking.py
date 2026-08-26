"""
OPTIONAL ADD-ON: Lightweight Wikidata Entity Linking.

Purpose: resolve a surface-form entity string (as produced by
entity_extraction.py, UNCHANGED) to a Wikidata QID, so that
entity_graph.py can recognize cross-language matches that pure
surface-form comparison misses -- e.g. claim entity "Bamangachi"
(English) and evidence entity "বামনগাছি" (Bengali) refer to the same
real-world place, but no amount of string normalization/suffix
stripping in entity_extraction.py can ever equate two different
scripts. Resolving both to the same QID (when Wikidata has it) fixes
exactly this gap.

DESIGN PRINCIPLES (all required by the "do not break the pipeline" brief):

1. STRICTLY ADDITIVE. This module does not change entity_extraction.py,
   does not change what "entities" are, and is not on the critical path
   of any existing stage. entity_graph.py calls it as an OPTIONAL extra
   signal (see linking failures below); nothing else in the pipeline
   imports this file.

2. NEVER RAISES. Every public function catches its own exceptions and
   returns None/{} on any failure (network error, timeout, rate limit,
   malformed response, missing dependency). Callers get either "here is
   a QID" or "no QID available" -- never an exception, exactly like the
   NER fallback pattern already used in entity_extraction.py and the
   translation fallback in query_generation.py.

3. CACHED. Wikidata lookups are network calls; a 1009-claim batch run
   would otherwise repeat the same lookups for common entities many
   times over. Results (including "no match found", cached as None) are
   persisted to a local JSON file so a re-run or a later claim never
   re-queries an entity string already looked up in this run or a past
   one. Cache is keyed by NORMALIZED entity text (via
   entity_extraction.normalize_entity_key), reusing the exact same
   normalization already applied for surface-form matching, so
   "Bamangachi" and "bamangachi station" (after suffix-stripping)
   resolve to the same cache entry as they would for string matching.

4. RATE-LIMITED / BOUNDED, WITH A CIRCUIT BREAKER (2026-07 fix). Two
   layers protect against Wikidata's request throttling:
     a) A minimum spacing is enforced between consecutive LIVE requests
        (MIN_REQUEST_INTERVAL_SECONDS), so a batch run doesn't fire
        lookups back-to-back and trip the throttle in the first place.
     b) If Wikidata responds with HTTP 429 anyway, a CIRCUIT BREAKER
        trips: live lookups are disabled for the rest of the process
        (all subsequent calls return None immediately, no network hit,
        no cache write), and exactly ONE warning is printed instead of
        one per entity. Without this, a single rate-limit event floods
        the log with a 429 for every remaining entity across every
        remaining claim in the batch, while the pipeline is still
        correctly falling back to surface-form-only matching regardless.
   429 results are NEVER written to the persistent cache (only genuine
   "no match found" or successful QIDs are) -- a 429 means "we don't
   know yet", not "this entity has no Wikidata page", so a future run
   (fresh circuit breaker) should still be free to try again.

5. NO NEW REQUIRED DEPENDENCY. Uses `requests`, which is already a
   pipeline dependency (see article_fetcher.py). Wikidata's public
   `wbsearchentities` REST endpoint needs no API key.
"""
import json
import threading
import time
from pathlib import Path

import requests

from . import config
from . import entity_extraction

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
LINKING_TIMEOUT_SECONDS = 5
MAX_LIVE_LOOKUPS_PER_RUN = 2000  # safety cap; cache keeps real usage far below this
MIN_REQUEST_INTERVAL_SECONDS = 1.0  # spacing between live Wikidata calls

_CACHE_PATH = config.OUTPUT_DIR / "wikidata_qid_cache.json"
_cache = None
_cache_lock = threading.Lock()
_live_lookup_count = 0
_last_request_time = 0.0

# Circuit breaker: once Wikidata returns a 429, stop making live calls
# for the rest of this process. Flipped exactly once; the warning that
# announces it is also printed exactly once (see _query_wikidata).
_circuit_open = False
_circuit_warning_printed = False

# Map our four dataset language tags -> Wikidata/MediaWiki language codes,
# used as search hints only (Wikidata search still works without this,
# but a correct language hint improves match quality for non-English
# surface forms).
_LANG_TO_WD = {
    "english": "en",
    "hindi": "hi",
    "bengali": "bn",
    "codemix": "en",  # best-effort default; codemix entities are often
                       # English-script proper nouns anyway
}


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    try:
        if _CACHE_PATH.exists():
            with open(_CACHE_PATH, encoding="utf-8") as f:
                _cache = json.load(f)
        else:
            _cache = {}
    except Exception as e:
        print(f"[entity_linking] WARNING: could not load QID cache ({e}); starting empty.")
        _cache = {}
    return _cache


def _save_cache() -> None:
    """Best-effort persist. A failed save never breaks the pipeline --
    the run just loses caching benefit for next time, nothing more."""
    try:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[entity_linking] WARNING: could not save QID cache ({e}).")


_RATE_LIMITED = object()  # sentinel: distinct from None ("no match found")


def _wait_for_rate_limit() -> None:
    """Blocks just long enough to keep consecutive live requests spaced
    at least MIN_REQUEST_INTERVAL_SECONDS apart. Cheap and simple --
    good enough to stay under Wikidata's anonymous-tier throttle for a
    single-process batch run; this is not meant to be a general-purpose
    rate limiter."""
    global _last_request_time
    now = time.monotonic()
    elapsed = now - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
    _last_request_time = time.monotonic()


def _query_wikidata(entity_text: str, lang_hint: str):
    """Single live lookup against Wikidata's wbsearchentities endpoint.
    Returns:
      - a QID string (e.g. 'Q12345') for the top result
      - None if the request succeeded but found no match
      - _RATE_LIMITED if the request failed specifically with HTTP 429
        (caller uses this to trip the circuit breaker and must NOT
        cache it as a genuine negative result)
    NEVER raises."""
    _wait_for_rate_limit()
    try:
        resp = requests.get(
            WIKIDATA_API_URL,
            params={
                "action": "wbsearchentities",
                "search": entity_text,
                "language": lang_hint,
                "format": "json",
                "limit": 1,
                "type": "item",
            },
            timeout=LINKING_TIMEOUT_SECONDS,
            headers={"User-Agent": "FIRE2026-Subtask2-Pipeline/1.0"},
        )
        if resp.status_code == 429:
            return _RATE_LIMITED
        resp.raise_for_status()
        data = resp.json()
        results = data.get("search", [])
        if not results:
            return None
        return results[0].get("id")  # e.g. "Q12345"
    except Exception as e:
        print(f"[entity_linking] WARNING: Wikidata lookup failed for "
              f"{entity_text!r} ({e}); treating as unlinked.")
        return None


def link_entity(entity_text: str, lang: str = "english") -> str | None:
    """Resolve one surface-form entity string to a Wikidata QID.

    Returns None if:
      - the entity is empty/too short after normalization
      - nothing is found on Wikidata
      - the lookup fails for any reason (network, timeout, malformed response)
      - the per-run live-lookup safety cap has been reached
      - the circuit breaker is open (Wikidata already rate-limited this
        process once -- see module docstring)

    Callers (entity_graph.py) MUST treat None as "no QID available" and
    fall back to existing surface-form matching -- this function guarantees
    it never raises, so a try/except at the call site is a defense-in-depth
    belt-and-braces measure, not a strict requirement.
    """
    global _live_lookup_count, _circuit_open, _circuit_warning_printed

    key = entity_extraction.normalize_entity_key(entity_text)
    if not key or len(key) < 2:
        return None

    cache = _load_cache()
    with _cache_lock:
        if key in cache:
            return cache[key]  # may legitimately be None (cached "no match")

        if _circuit_open:
            # Already rate-limited once this run -- don't even try, so
            # we don't spam the log or waste time on doomed requests for
            # the rest of a large batch.
            return None

        if _live_lookup_count >= MAX_LIVE_LOOKUPS_PER_RUN:
            # Safety cap hit -- degrade to "unlinked" rather than block
            # or crash a large batch run. Not cached, so a future run
            # (with the cap reset) can still try.
            return None
        _live_lookup_count += 1

    lang_hint = _LANG_TO_WD.get((lang or "").lower().strip(), "en")
    result = _query_wikidata(entity_text, lang_hint)

    if result is _RATE_LIMITED:
        with _cache_lock:
            _circuit_open = True
            if not _circuit_warning_printed:
                _circuit_warning_printed = True
                print(f"[entity_linking] WARNING: Wikidata rate-limited this "
                      f"process (HTTP 429) while looking up {entity_text!r}. "
                      f"Disabling further live QID lookups for the rest of this "
                      f"run -- falling back to surface-form-only entity matching "
                      f"from here on. This is NOT cached, so a future run will "
                      f"try again fresh.")
        return None  # this call's result -- unlinked, not cached below

    qid = result

    with _cache_lock:
        cache[key] = qid
        _save_cache()

    return qid


def link_entities(entities: list, lang: str = "english") -> dict:
    """Batch convenience wrapper: {entity_text: qid_or_None} for a list
    of surface-form entity strings. Never raises -- any individual
    lookup failure just yields None for that entity."""
    out = {}
    for e in entities:
        try:
            out[e] = link_entity(e, lang=lang)
        except Exception as ex:
            # Belt-and-braces: even if link_entity's own guards somehow
            # miss something, this stage must never take down
            # entity_graph.py's caller.
            print(f"[entity_linking] WARNING: unexpected error linking {e!r} ({ex}); skipping.")
            out[e] = None
    return out


if __name__ == "__main__":
    # Smoke test -- real cross-language pair from the pipeline's own
    # documented failure case (see entity_extraction.py docstring).
    test_pairs = [
        ("Bamangachi", "english"),
        ("বামনগাছি", "bengali"),
    ]
    for text, lang in test_pairs:
        qid = link_entity(text, lang=lang)
        print(f"  {text!r} ({lang}) -> {qid}")