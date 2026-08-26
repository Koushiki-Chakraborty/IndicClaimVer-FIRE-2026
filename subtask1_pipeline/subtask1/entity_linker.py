import json
import re
import time
import unicodedata
from pathlib import Path

from SPARQLWrapper import JSON, SPARQLWrapper

import config

_cache = None


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if config.SPARQL_CACHE_PATH.exists():
        with open(config.SPARQL_CACHE_PATH, encoding="utf-8") as f:
            _cache = json.load(f)
    else:
        _cache = {}
    return _cache


def _save_cache():
    with open(config.SPARQL_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False)


def _sparql_query(query: str) -> list:
    sparql = SPARQLWrapper(config.SPARQL_ENDPOINT, agent=config.SPARQL_USER_AGENT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    # A hung/slow Wikidata response with no timeout blocks this call (and the
    # entire entity-linking loop behind it) forever -- observed directly: a
    # real smoketest run froze for 1h40m+ on one query with no exception ever
    # raised. Every caller already wraps this in try/except and degrades
    # gracefully on failure (see get_candidates_exact/_check_type_membership),
    # so a bounded timeout just turns an infinite hang into that same,
    # already-handled failure path.
    sparql.setTimeout(config.SPARQL_TIMEOUT_SECONDS)
    results = sparql.query().convert()
    time.sleep(config.SPARQL_SLEEP_SECONDS)  # politeness delay, every call
    return results["results"]["bindings"]


# ---------------------------------------------------------------------
# NER -- with a heuristic fallback if the model can't load
# ---------------------------------------------------------------------
_ner_pipeline = None
_ner_load_failed = False


def _get_ner():
    global _ner_pipeline, _ner_load_failed
    if _ner_pipeline is not None or _ner_load_failed:
        return _ner_pipeline
    try:
        from transformers import pipeline
        _ner_pipeline = pipeline(
            "token-classification",
            model=config.NER_MODEL,
            aggregation_strategy="simple",
        )
        print(f"[entity_linker] Loaded NER model: {config.NER_MODEL}")
    except Exception as e:
        print(f"[entity_linker] WARNING: could not load {config.NER_MODEL} ({e}). "
              f"Falling back to heuristic entity extraction. Verify the model "
              f"name loads on your machine (Gate 1) for real NER quality.")
        _ner_load_failed = True
        _ner_pipeline = None
    return _ner_pipeline


def _heuristic_entities(text: str) -> list:
    """Fallback: capitalized multi-word sequences (English/code-mixed)
    plus long Devanagari/Bengali tokens (proper-noun-length heuristic).
    Much lower recall/precision than a real NER model -- only meant to
    keep the pipeline running, not as a final design choice."""
    entities = []
    for m in re.finditer(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b", text):
        entities.append(m.group())
    for m in re.finditer(r"[\u0900-\u097F]{4,}|[\u0980-\u09FF]{4,}", text):
        entities.append(m.group())
    # de-dup, preserve order
    seen = set()
    out = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _is_word_char(ch: str) -> bool:
    """Unicode-aware word-character check. Crucially includes combining
    marks (category Mn/Mc) -- Devanagari/Bengali virama, matras, and nukta
    are marks, not letters, so a plain isalnum() check would treat them as
    word BOUNDARIES and reproduce the exact corruption this is fixing."""
    cat = unicodedata.category(ch)
    return cat[0] in ("L", "N") or cat in ("Mn", "Mc")


def _expand_to_word_boundary(text: str, start: int, end: int) -> tuple:
    """Grows [start, end) outward to the nearest real word boundary in the
    ORIGINAL text. Fixes sub-word fragmentation (e.g. NER tagging just
    "Kam" out of "Kamrup") without needing to merge separate NER spans --
    two adjacent fragments of the same true word independently expand to
    the same full span, so a later dedupe step collapses them into one."""
    while start > 0 and _is_word_char(text[start - 1]):
        start -= 1
    while end < len(text) and _is_word_char(text[end]):
        end += 1
    return start, end


def extract_entities(text: str) -> list:
    """Returns list of surface mention strings.

    BUG FIX 2026-07-12: mentions used to come from the NER pipeline's own
    `word` field, which is reconstructed from sub-word tokens and can be
    corrupted -- dropped Devanagari/Bengali diacritics, lowercased English,
    or truncated at sub-word boundaries for unfamiliar proper nouns
    (verified: "rag"/"kam"/"sil" instead of "Raghvendra"/"Kamrup"/"Silsako";
    "delhi" instead of "Delhi"; "दिलली" instead of "दिल्ली"). The model's
    `start`/`end` character offsets remain reliable even when `word` isn't,
    across Hindi/Bengali/English -- so mentions are now re-sliced from the
    original text using those offsets (expanded to the containing word's
    real boundaries) instead of trusting the model's own reconstruction.
    """
    ner = _get_ner()
    if ner is None:
        return _heuristic_entities(text)
    try:
        truncated = text[:512]  # guard against very long inputs
        results = ner(truncated)
        seen = set()
        out = []
        for r in results:
            start, end = _expand_to_word_boundary(truncated, r["start"], r["end"])
            mention = truncated[start:end].strip()
            if len(mention) <= 1 or mention in seen:
                continue
            seen.add(mention)
            out.append(mention)
        return out
    except Exception as e:
        print(f"[entity_linker] NER inference failed ({e}), using heuristic fallback")
        return _heuristic_entities(text)


# ---------------------------------------------------------------------
# Wikidata entity linking
# ---------------------------------------------------------------------
def link_entity(label: str, lang: str = "en") -> str | None:
    """Returns a QID string (e.g. 'Q3737857') or None. Tries exact label
    match first, then alt-label match for better recall on informal
    mentions (e.g. 'Kejriwal' vs 'Arvind Kejriwal').

    Disambiguation: many labels match multiple Wikidata items (common
    names, generic org names), and Wikidata's SPARQL engine gives no
    ordering guarantee -- a bare `LIMIT 1` picks an arbitrary match, which
    could silently link to an obscure/wrong entity. We rank candidates by
    wikibase:sitelinks (how many Wikipedia language editions have a page
    for it) as a cheap, standard notability signal, so a common name
    resolves to the well-known entity rather than a random namesake.
    OPTIONAL keeps candidates with zero sitelinks in the running (ranked
    last) instead of excluding them outright -- obscure but real entities
    (e.g. minor local officials) can still resolve correctly."""
    cache = _load_cache()
    key = f"link::{lang}::{label}"
    if key in cache:
        return cache[key]

    lang = lang if lang in ("hi", "bn", "en") else "en"
    qid = _link_entity_query(label, lang)

    # DEFENSIVE FALLBACK 2026-07-12: English is the only one of these three
    # languages with letter case. extract_entities() now preserves the
    # source text's original casing (verified: recovers "Delhi"/"BJP"
    # correctly), so this shouldn't often be needed -- but if the
    # exact-cased label still finds nothing, try a title-cased variant
    # before giving up, in case the source text itself was written in an
    # unusual case.
    if qid is None and lang == "en" and label != label.title():
        qid = _link_entity_query(label.title(), lang)

    cache[key] = qid
    _save_cache()
    return qid


def _link_entity_query(label: str, lang: str) -> str | None:
    query = f"""
    SELECT ?item WHERE {{
      {{ ?item rdfs:label "{_escape(label)}"@{lang}. }}
      UNION
      {{ ?item skos:altLabel "{_escape(label)}"@{lang}. }}
      OPTIONAL {{ ?item wikibase:sitelinks ?sitelinks. }}
    }} ORDER BY DESC(?sitelinks) LIMIT 1
    """
    try:
        bindings = _sparql_query(query)
        if bindings:
            return bindings[0]["item"]["value"].split("/")[-1]
        return None
    except Exception as e:
        print(f"[entity_linker] SPARQL link_entity failed for '{label}' ({e})")
        return None


def _escape(s: str) -> str:
    return s.replace('"', '\\"')


def get_entity_label_en(qid: str) -> str | None:
    cache = _load_cache()
    key = f"label::{qid}"
    if key in cache:
        return cache[key]
    query = f"""
    SELECT ?label WHERE {{
      wd:{qid} rdfs:label ?label.
      FILTER(LANG(?label) = "en")
    }} LIMIT 1
    """
    try:
        bindings = _sparql_query(query)
        label = bindings[0]["label"]["value"] if bindings else None
        cache[key] = label
        _save_cache()
        return label
    except Exception as e:
        print(f"[entity_linker] SPARQL get_entity_label_en failed for {qid} ({e})")
        return None


def get_temporal_facts(qid: str, property_id: str = "P39") -> list:
    """Fetch structured temporal qualifiers for a Wikidata property.
    Default P39 = 'position held'. Returns a list of dicts:
        {predicate, object_label, valid_from, valid_to}
    valid_from/valid_to are None when Wikidata has no qualifier for
    that statement -- this is expected and must be handled by callers,
    not treated as an error.
    """
    cache = _load_cache()
    key = f"temporal::{qid}::{property_id}"
    if key in cache:
        return cache[key]

    query = f"""
    SELECT ?valueLabel ?start ?end WHERE {{
      wd:{qid} p:{property_id} ?statement .
      ?statement ps:{property_id} ?value .
      OPTIONAL {{ ?statement pq:P580 ?start . }}
      OPTIONAL {{ ?statement pq:P582 ?end . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    try:
        bindings = _sparql_query(query)
        facts = []
        for b in bindings:
            facts.append({
                "predicate": property_id,
                "object_label": b.get("valueLabel", {}).get("value"),
                "valid_from": b.get("start", {}).get("value"),  # None if absent
                "valid_to": b.get("end", {}).get("value"),
            })
        cache[key] = facts
        _save_cache()
        return facts
    except Exception as e:
        print(f"[entity_linker] SPARQL get_temporal_facts failed for {qid} ({e})")
        return []


if __name__ == "__main__":
    # Smoke test -- run this FIRST on your own machine (Gate 1).
    # This sandbox could not reach query.wikidata.org to verify these
    # calls actually work (confirmed via curl: 403 on the egress proxy).
    print("Testing entity extraction (heuristic or model, whichever loads)...")
    test_text = "Arvind Kejriwal was arrested by the CBI in the excise policy case."
    ents = extract_entities(test_text)
    print("Entities:", ents)

    print("\nTesting Wikidata linking for 'Arvind Kejriwal'...")
    qid = link_entity("Arvind Kejriwal", "en")
    print("QID:", qid)

    if qid:
        print(f"\nTesting temporal facts (P39, position held) for {qid}...")
        facts = get_temporal_facts(qid, "P39")
        for f in facts:
            print(" ", f)
