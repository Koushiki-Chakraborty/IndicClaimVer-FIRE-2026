"""
STEP 2: Query Generation.

For every claim, regardless of language, we generate:
  - "english_translation": the claim translated to English
  - "native_query": a short search query in the claim's ORIGINAL language
  - "english_query": a short search query in English

Why both native + English: regional-language news sources (Bengali/Hindi)
are often only indexed in their own language. English-only search misses
them. See the Bamangachi test case for a concrete example of this.

DETERMINISTIC IMPLEMENTATION (2026-07-18 rewrite): this stage used to
call Gemini. It now uses no LLM at all:

  1. Entities are pulled from the claim text using
     entity_extraction.extract_entities() -- reused UNCHANGED from the
     existing pipeline module, so claim-entity extraction here stays
     identical to what the entity-graph ranking stage already does.
  2. Stopwords are stripped from the raw claim tokens using
     `stopwordsiso` (covers "en", "hi", "bn"; "codemix" claims union
     all three since they mix scripts within one sentence).
  3. "native_query" = entities (kept as intact multi-word spans) +
     remaining non-stopword keywords, in order of appearance, capped
     at a short word count -- mirrors the old prompt's "5-8 words,
     keep key entities, drop filler words" instruction.
  4. "english_translation" comes from `deep-translator`'s
     GoogleTranslator (free, no API key). If translation fails for ANY
     reason (network, rate limit, unsupported input, library missing),
     we fall back to the raw claim text -- see _translate_to_english()
     -- so this stage can never crash the pipeline.
  5. "english_query" is built the same way as native_query, but from
     the translated (or, on fallback, raw) text.

Public function name + signature + return keys are UNCHANGED, so
pipeline.py (and everything downstream) needs zero modification:

    generate_queries(claim_text: str, lang: str) -> dict
    -> {"english_translation": ..., "native_query": ..., "english_query": ...}
"""
import re

from . import entity_extraction

try:
    from deep_translator import GoogleTranslator
    _TRANSLATOR_AVAILABLE = True
except ImportError:
    _TRANSLATOR_AVAILABLE = False

try:
    import stopwordsiso as _stopwordsiso
except ImportError:
    _stopwordsiso = None

REQUIRED_KEYS = ("english_translation", "native_query", "english_query")

# Map our four dataset language tags -> stopwordsiso ISO codes.
# "codemix" claims mix English/Hindi/Bengali script within a single
# sentence, so we union all three stopword sets rather than picking one.
_LANG_TO_ISO = {
    "english": ("en",),
    "hindi": ("hi",),
    "bengali": ("bn",),
    "codemix": ("en", "hi", "bn"),
}

_MAX_QUERY_WORDS = 8

# \w is Unicode-aware for str patterns in Python 3, so this tokenizes
# Latin, Devanagari, and Bengali text alike without extra flags.
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# OPTIONAL ADD-ON (2026-07): matches tokens that are (or contain) a
# number -- e.g. "27,561", "89%", "24,383" -- optionally with a trailing
# '%'. Diagnosed directly from dev-set F1 errors: claims stating a
# specific statistic (Covid case counts, percentages) were frequently
# retrieved against a DIFFERENT day's/report's numbers, because the
# exact figure was getting crowded out of the 8-word query cap by
# generic keywords ("Delhi", "Covid", "cases", "new") that also appear
# in every other day's article on the same running topic. Preserving
# the literal number(s) in the query is a cheap, targeted way to nudge
# retrieval/BM25 toward the specific article that actually contains
# that figure, rather than a topically-similar one with a different date.
_NUMERIC_RE = re.compile(r"\d[\d,.]*%?")


def _stopword_set(lang: str) -> set:
    """Lowercased stopword set for the given claim language. Falls back
    to an empty set if stopwordsiso isn't installed, and to English if
    `lang` isn't one of our four known tags -- never raises."""
    if _stopwordsiso is None:
        return set()
    codes = _LANG_TO_ISO.get((lang or "").lower().strip(), ("en",))
    words = set()
    for code in codes:
        try:
            words |= {w.lower() for w in _stopwordsiso.stopwords(code)}
        except Exception:
            continue
    return words


def _tokenize(text: str) -> list:
    """Unicode-aware whitespace/punctuation tokenizer."""
    return _WORD_RE.findall(text or "")


def _build_query(text: str, entities: list, stopwords: set,
                  max_words: int = _MAX_QUERY_WORDS) -> str:
    """Deterministic short-query builder: NUMERIC TOKENS FIRST (verbatim,
    e.g. exact case counts/percentages -- see _NUMERIC_RE above), then
    entities (kept intact as multi-word spans -- they carry the most
    search-relevant signal), then remaining non-stopword keywords in
    order of appearance, deduplicated case-insensitively, capped at
    max_words. Never returns an empty string as long as `text` has any
    content.

    Numeric tokens are added FIRST (ahead of entities) because they were
    the most frequent casualty of the word cap in practice -- an 8-word
    budget filled by "Delhi Covid cases new record" left no room for the
    one token ("27,561") that actually pins down WHICH day's article is
    being described, and that's exactly the recurring dev-set error
    pattern this addresses.
    """
    picked = []
    seen = set()          # phrase-level keys (blocks exact re-adds)
    seen_words = set()    # individual-word keys (blocks entity words from
                           # also being re-added piecemeal as keywords)

    def _add(tok: str):
        key = tok.lower()
        if not tok.strip() or key in seen:
            return
        seen.add(key)
        seen_words.update(_WORD_RE.findall(key))
        picked.append(tok)

    # OPTIONAL ADD-ON: numeric tokens first, verbatim, exempt from the
    # stopword/length filter that would otherwise apply to keywords below
    # (a bare number like "89" or "5" would normally be too short/generic
    # to keep, but here it's exactly the disambiguating detail we want).
    for tok in _NUMERIC_RE.findall(text or ""):
        if len(picked) >= max_words:
            break
        _add(tok)

    for ent in entities:
        if len(picked) >= max_words:
            break
        _add(ent.strip())

    if len(picked) < max_words:
        for tok in _tokenize(text):
            if len(picked) >= max_words:
                break
            if len(tok) <= 1 or tok.lower() in stopwords or tok.lower() in seen_words:
                continue
            _add(tok)

    if not picked:
        # Last-resort fallback (e.g. text is entirely stopwords/short
        # tokens) so we never hand an empty query downstream to retrieval.
        picked = _tokenize(text)[:max_words] or [text.strip()[:40]]

    return " ".join(picked)


def _translate_to_english(claim_text: str) -> tuple:
    """Translate via deep-translator's GoogleTranslator (no API key,
    no LLM). Returns (text, succeeded). On ANY failure -- library
    missing, network error, empty response -- returns the raw claim
    text unchanged with succeeded=False, so callers can degrade
    gracefully instead of crashing."""
    if not _TRANSLATOR_AVAILABLE:
        return claim_text, False
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(claim_text)
        if translated and translated.strip():
            return translated.strip(), True
        return claim_text, False
    except Exception as e:
        print(f"[query_generation] WARNING: translation failed ({e}); "
              f"falling back to raw claim text for english_translation.")
        return claim_text, False


def generate_queries(claim_text: str, lang: str) -> dict:
    """Returns {"english_translation": str, "native_query": str, "english_query": str}

    Fully deterministic, no LLM call. Reuses
    entity_extraction.extract_entities() unchanged for both the
    native-language claim text and the translated (or fallback)
    English text.
    """
    native_stopwords = _stopword_set(lang)
    native_entities = entity_extraction.extract_entities(claim_text)
    native_query = _build_query(claim_text, native_entities, native_stopwords)

    english_translation, translation_succeeded = _translate_to_english(claim_text)

    # If translation actually succeeded, the resulting text is genuine
    # English -- use English stopwords regardless of the claim's
    # original language. If it fell back to the raw claim text, keep
    # using the claim's own stopword set (translation didn't happen,
    # so the text is still in its original language/script).
    english_stopwords = _stopword_set("english") if translation_succeeded else native_stopwords
    english_entities = entity_extraction.extract_entities(english_translation)
    english_query = _build_query(english_translation, english_entities, english_stopwords)

    return {
        "english_translation": english_translation,
        "native_query": native_query,
        "english_query": english_query,
    }


if __name__ == "__main__":
    # Smoke test with the real Bamangachi claim
    import json
    test_claim = "লকডাউনে শুনশান বামনগাছি স্টেশনই দুষ্কৃতীদের স্বর্গরাজ্য, নিয়মিত বসছে মদের আসর!"
    result = generate_queries(test_claim, "bengali")
    print(json.dumps(result, ensure_ascii=False, indent=2))