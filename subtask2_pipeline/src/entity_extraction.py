"""
STEP 2b: Entity Extraction (Claim Preprocessing + Evidence Preprocessing).

Breaks claim text AND evidence snippet text into entities, so a later
ranking stage can score evidence by how many entities it shares with
the claim (the "entity graph" idea) -- e.g. an evidence snippet
mentioning "Bamangachi" should score higher for the Bamangachi claim
than one that only shares generic words like "lockdown".

This module does NOT do Wikidata/QID linking -- that's a KG-stage
concern (cross-language identity resolution, e.g. matching "Bamangachi"
to "বামনগাছি" as the same real-world entity). This module only extracts
SURFACE-FORM entities from raw text.

BUG FIX (2026-07-16): entity_overlap_count() used to require EXACT
string match after lowercasing. Real test data showed this fails on
Bengali/Hindi due to inflectional suffixes -- e.g. the claim's
"বামনগাছি স্টেশনই" (with the "-ই" emphatic/case marker) did NOT match
evidence text's "বামনগাছি স্টেশন" (bare form), even though it's
unambiguously the same real-world entity (Bamangachi station), and
that evidence item was genuinely on-topic. Fixed by (1) stripping a
list of common Bengali/Hindi trailing suffixes before comparing, and
(2) falling back to substring containment (either normalized string
contains the other) when exact match still fails after normalization.
This is a heuristic, not a linguistic analyzer -- it will not catch
every morphological variant, but it fixes the specific failure mode
observed and is a meaningful improvement over exact-match-only.

Falls back to a lightweight heuristic extractor if the NER model can't
load, so this never crashes the pipeline -- just degrades gracefully.
"""
import re
import unicodedata

import src.config as config

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
        print(f"[entity_extraction] Loaded NER model: {config.NER_MODEL}")
    except Exception as e:
        print(f"[entity_extraction] WARNING: could not load {config.NER_MODEL} ({e}). "
              f"Falling back to heuristic entity extraction.")
        _ner_load_failed = True
        _ner_pipeline = None
    return _ner_pipeline


def _is_word_char(ch: str) -> bool:
    """Unicode-aware word-character check, including combining marks
    (Devanagari/Bengali matras/virama/nukta are marks, not letters --
    isalnum() alone would wrongly treat them as word boundaries)."""
    cat = unicodedata.category(ch)
    return cat[0] in ("L", "N") or cat in ("Mn", "Mc")


def _expand_to_word_boundary(text: str, start: int, end: int) -> tuple:
    while start > 0 and _is_word_char(text[start - 1]):
        start -= 1
    while end < len(text) and _is_word_char(text[end]):
        end += 1
    return start, end


def _heuristic_entities(text: str) -> list:
    """Fallback: capitalized multi-word sequences (English/code-mixed)
    plus long Devanagari/Bengali tokens (proper-noun-length heuristic).
    Lower precision/recall than a real NER model -- keeps the pipeline
    running, not a final design choice."""
    entities = []
    for m in re.finditer(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b", text):
        entities.append(m.group())
    for m in re.finditer(r"[\u0900-\u097F]{4,}|[\u0980-\u09FF]{4,}", text):
        entities.append(m.group())
    seen = set()
    out = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def extract_entities(text: str) -> list:
    """Returns a list of surface-form entity strings extracted from text.
    Works on claim text OR evidence snippet text -- same function, same
    logic, so claim-entities and evidence-entities are directly comparable."""
    ner = _get_ner()
    if ner is None:
        return _heuristic_entities(text)

    try:
        truncated = text[:512]
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
        print(f"[entity_extraction] NER inference failed ({e}), using heuristic fallback")
        return _heuristic_entities(text)


# ---------------------------------------------------------------------
# Entity normalization + overlap scoring
# ---------------------------------------------------------------------

# Common Bengali/Hindi trailing suffixes (case markers, emphatic particles,
# postpositions) that attach to nouns and change the surface form without
# changing the underlying entity. Ordered longest-first so a longer
# suffix is stripped before a shorter one that might be its substring.
_BN_HI_SUFFIXES = sorted([
    "দের", "গুলো", "গুলি", "টির", "টার",  # Bengali: plural/possessive/definite
    "তে", "কে", "য়ের", "য়ে", "টি", "টা", "ই", "র", "ে",  # Bengali: case/emphatic
    "ों", "में", "से", "को", "की", "के", "का", "है",  # Hindi: case markers/copula
], key=len, reverse=True)


def _normalize_entity(entity: str) -> str:
    """Lowercase (for English/codemix) + strip trailing Bengali/Hindi
    suffixes so inflected forms of the same word normalize to the same
    string. Heuristic, not a real morphological analyzer."""
    normalized = entity.strip().lower()
    for suffix in _BN_HI_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
            normalized = normalized[: -len(suffix)]
            break  # only strip one suffix layer -- avoid over-stripping
    return normalized.strip()


def normalize_entity_key(entity: str) -> str:
    """Public wrapper around _normalize_entity(), for callers outside
    this module (e.g. entity_graph_kg.py, which needs a stable,
    normalized string to use as a Neo4j Entity node key). Keeps the
    normalization logic defined in exactly one place."""
    return _normalize_entity(entity)


def _entities_match(a: str, b: str) -> bool:
    """True if two entity strings likely refer to the same real-world
    entity: exact match after normalization, OR one normalized form
    contains the other (handles cases the suffix list doesn't cover,
    e.g. multi-word spans where only part matches)."""
    na, nb = _normalize_entity(a), _normalize_entity(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # Substring containment -- guard against tiny/generic strings
    # matching everything (require at least 3 chars to compare this way)
    if len(na) >= 3 and len(nb) >= 3 and (na in nb or nb in na):
        return True
    return False


def entity_overlap_count(claim_entities: list, evidence_entities: list) -> int:
    """Counts how many claim entities have at least one match among the
    evidence entities, using normalized + substring-containment matching
    (see _entities_match) instead of requiring exact string equality.

    BUG FIX: previously used exact-match-after-lowercase set intersection,
    which failed on real Bengali data (see module docstring)."""
    count = 0
    matched_evidence_idx = set()
    for ce in claim_entities:
        for i, ee in enumerate(evidence_entities):
            if i in matched_evidence_idx:
                continue
            if _entities_match(ce, ee):
                count += 1
                matched_evidence_idx.add(i)
                break
    return count


if __name__ == "__main__":
    # Smoke test on the REAL failure case found in production output:
    # claim entity "বামনগাছি স্টেশনই" vs evidence entity "বামনগাছি স্টেশন"
    # (same entity, different Bengali suffix) -- previously scored 0, should now score 1.
    claim_entities = ["বামনগাছি স্টেশনই"]
    evidence_entities_should_match = ["বামনগাছি স্টেশন"]  # from real evidence #3
    evidence_entities_should_not_match = ["Haryana", "World", "Anil Vij"]  # from real evidence #7

    print("Testing the real bug found in production run:")
    print(f"  claim entity: {claim_entities}")
    print(f"  evidence #3 entities (should now match): {evidence_entities_should_match}")
    print(f"  -> overlap count: {entity_overlap_count(claim_entities, evidence_entities_should_match)} (expected: 1)")
    print(f"  evidence #7 entities (should NOT match): {evidence_entities_should_not_match}")
    print(f"  -> overlap count: {entity_overlap_count(claim_entities, evidence_entities_should_not_match)} (expected: 0)")

    print("\nTesting English extraction (regression check on original example):")
    claim_text = "During lockdown, the deserted Bamangachi station has become a haven for criminals, with liquor parties held regularly"
    evidence_text_relevant = "Top Lounges in Bamangachi - Book a Table in Lounge Bars"
    evidence_text_irrelevant = "During lockdown a liquor party was going on inside a school in Tufanganj"

    claim_ents = extract_entities(claim_text)
    ev1_ents = extract_entities(evidence_text_relevant)
    ev2_ents = extract_entities(evidence_text_irrelevant)
    print("  claim entities:", claim_ents)
    print("  overlap with relevant evidence:", entity_overlap_count(claim_ents, ev1_ents), "(expected: >=1)")
    print("  overlap with irrelevant evidence:", entity_overlap_count(claim_ents, ev2_ents), "(expected: 0)")