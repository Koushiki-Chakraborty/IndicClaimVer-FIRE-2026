import hashlib
import json
import os
import pickle
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

# Parallel KG build support: when running multiple shard processes
# concurrently against the same production database (see
# subtask1/scripts/kg_pipeline.py's `shard` stage), each process's disk caches
# (candidate/type/LLM-alias/embedding lookups) must NOT share a single
# file -- these caches are plain json.dump()/pickle.dump() writes, not
# atomic, so two processes writing the same file at the same time risks
# corrupting it. Set CACHE_SHARD_SUFFIX (e.g. "shard0") to give each
# process its own isolated cache files instead. Empty by default --
# a single sequential run is unaffected.
_CACHE_SHARD_SUFFIX = os.getenv("CACHE_SHARD_SUFFIX", "")


def _shard_cache_path(path: Path) -> Path:
    if not _CACHE_SHARD_SUFFIX:
        return path
    return path.with_name(f"{path.stem}.{_CACHE_SHARD_SUFFIX}{path.suffix}")


def _merged_cache_path(path: Path) -> Path:
    """Shared, read-only-at-runtime cache combining every shard's discoveries
    as of the last manual merge (see subtask1/scripts/merge_shard_caches.py).
    Loaded IN ADDITION to this process's own shard-specific file so that
    "what every shard has ever found" and "what this shard has found since
    the last merge" are both available -- never just one or the other.
    Still never written to by a running shard (each shard keeps saving to
    its own isolated file, exactly as before) -- only the merge script
    writes here, and only while no shard is running. This is what makes it
    safe to read from concurrently: it's frozen at load time, not a live
    shared file multiple processes write to."""
    return path.with_name(f"{path.stem}.merged{path.suffix}")
from subtask1 import entity_linker, data_loader
from subtask1.neo4j_client import Neo4jClient

STATS = {
    "sparql_calls": 0, "sparql_cache_hits": 0,
    "wbsearch_calls": 0, "wbsearch_used_count": 0,
    "alias_used_count": 0, "morphology_used_count": 0,
    "code_mixed_used_count": 0, "coreference_used_count": 0,
    "llm_alias_used_count": 0,
    "llm_alias_trusted_count": 0, "llm_tier2_attempted_count": 0,
    "llm_tier2_success_count": 0, "wikipedia_resolver_used_count": 0,
    "resolution_source": Counter(),
}

INDIA_QID = "Q668"

# Type-aware candidate filtering roots. PER uses a direct P31=Q5 (human)
# check (every human item is directly instance-of human, no traversal
# needed). ORG/LOC instead check P31 THEN P279* (subclass-of, transitive)
# against a single root class each -- this covers every real subtype
# (political party, university, business, ... under organization; city,
# country, river, ... under geographic entity) via Wikidata's own class
# hierarchy, with no enumerated subtype list to maintain.
ORG_ROOT_QID = "Q43229"    # organization
LOC_ROOT_QID = "Q27096213"  # geographic entity

# ---------------------------------------------------------------------
# Bengali morphology (validated 4-rule set)
# ---------------------------------------------------------------------
MATRA_E = chr(0x09c7)
RA = chr(0x09b0)
TA = chr(0x09a4)
TE_SUFFIX = TA + MATRA_E
SAFE_SUFFIXES = [MATRA_E + RA, MATRA_E, RA]  # ের, ে, র


def _is_vowel_sign(ch):
    return unicodedata.category(ch) in ("Mn", "Mc")


def strip_bn_suffix(word: str) -> str | None:
    # vowel-final-stem locative (validated discriminator)
    if word.endswith(TE_SUFFIX) and len(word) - 2 >= 2:
        stem = word[:-2]
        if _is_vowel_sign(stem[-1]):
            return stem
    for suf in SAFE_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return word[:-len(suf)]
    return None


# ---------------------------------------------------------------------
# Candidate generation -- disk-persisted cache (not just in-memory),
# separate from entity_linker's own disk cache since it stores full
# candidate lists (with sitelinks/country/type), not single QIDs. This
# is what lets a Claim-only re-run reuse everything already fetched
# during the first pass instead of re-querying Wikidata from scratch.
# ---------------------------------------------------------------------
_CAND_CACHE_PATH = _shard_cache_path(Path(__file__).parent / "integrated_candidate_cache.json")
_cand_cache = {}


def _cand_cache_key_to_str(key):
    return "||".join(key)


def load_cand_cache():
    global _cand_cache
    merged_path = _merged_cache_path(_CAND_CACHE_PATH)
    if merged_path.exists():
        with open(merged_path, encoding="utf-8") as f:
            raw = json.load(f)
        _cand_cache = {tuple(k.split("||")): v for k, v in raw.items()}
        print(f"[integrated_pipeline] Loaded {len(_cand_cache)} cached candidate lookups from the shared merged cache", flush=True)
    if _CAND_CACHE_PATH.exists():
        with open(_CAND_CACHE_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        own = {tuple(k.split("||")): v for k, v in raw.items()}
        _cand_cache.update(own)  # this shard's own file wins on conflict -- most recent local knowledge
        print(f"[integrated_pipeline] Loaded {len(own)} cached candidate lookups from this shard's own disk cache "
              f"({len(_cand_cache)} total after merging with the shared cache)", flush=True)


def save_cand_cache():
    raw = {_cand_cache_key_to_str(k): v for k, v in _cand_cache.items()}
    with open(_CAND_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False)


_cand_cache_dirty = 0
_SAVE_CAND_CACHE_EVERY = 20


def _maybe_save_cand_cache():
    """save_cand_cache() rewrites the WHOLE dict every call -- fine at
    smoketest scale (hundreds of entries), but at real scale this dict
    grows into the tens of thousands, and a full-dict JSON dump on every
    single lookup turns into real, avoidable overhead across a
    multi-hour/day run (O(n) cost per write, over up to n lookups).
    Throttled to every _SAVE_CAND_CACHE_EVERY updates instead of every
    one -- worst case on an interruption, a handful of lookups get
    redone next time (cheap: cache-miss cost is a network call either
    way, this only changes how many of THOSE get repeated, not whether
    the pipeline is correct)."""
    global _cand_cache_dirty
    _cand_cache_dirty += 1
    if _cand_cache_dirty >= _SAVE_CAND_CACHE_EVERY:
        save_cand_cache()
        _cand_cache_dirty = 0


load_cand_cache()


def _cache_entry_is_fresh(cached: list) -> bool:
    """A cached candidate list from before the label/description fix has
    entries missing the 'label' key entirely (not just label=None -- that's
    a legitimate 'Wikidata has no English label' result). Treating a stale
    hit as a miss makes the cache self-healing: it stops silently serving
    pre-fix data forever, without needing a manual eviction script every
    time this class of fix happens again. Same reasoning for 'is_org' --
    added when ORG/LOC type-filtering was introduced; entries from before
    that lack it entirely and must not be silently read back as False."""
    return not cached or all("label" in c and "is_org" in c for c in cached)


# ---------------------------------------------------------------------
# ORG/LOC type flags -- kept OUT of the main candidate queries above.
# Embedding the P31/P279* check inline (as an OPTIONAL alongside the
# label search or the sitelinks/country/type VALUES query) was measured
# at 43-66s per call -- Blazegraph loses its optimized handling of the
# chained wdt:P31/wdt:P279* path the moment it competes with another
# OPTIONAL/property-path block in the same query. Run alone, in its own
# query, that same chained path takes ~14s for a batch of QIDs. So this
# runs as a separate, QID-keyed, disk-cached step: cost is paid once per
# distinct QID ever seen (not once per mention), and never again after.
# ---------------------------------------------------------------------
_TYPE_CACHE_PATH = _shard_cache_path(Path(__file__).parent / "integrated_type_cache.json")
_type_cache = {}


def _load_type_cache():
    global _type_cache
    merged_path = _merged_cache_path(_TYPE_CACHE_PATH)
    if merged_path.exists():
        with open(merged_path, encoding="utf-8") as f:
            _type_cache = json.load(f)
        print(f"[integrated_pipeline] Loaded {len(_type_cache)} cached type-flag lookups from the shared merged cache", flush=True)
    if _TYPE_CACHE_PATH.exists():
        with open(_TYPE_CACHE_PATH, encoding="utf-8") as f:
            own = json.load(f)
        _type_cache.update(own)
        print(f"[integrated_pipeline] Loaded {len(own)} cached type-flag lookups from this shard's own disk cache "
              f"({len(_type_cache)} total after merging with the shared cache)", flush=True)


def save_type_cache():
    with open(_TYPE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_type_cache, f, ensure_ascii=False)


_load_type_cache()


def _check_type_membership(qids: list, root_qid: str) -> tuple:
    """Which of `qids` are (transitively) P31/P279* wd:root_qid. One
    query, Wikidata's own optimized chained-path idiom -- do NOT split
    this into two triple patterns (?item wdt:P31 ?t. ?t wdt:P279* wd:root)
    or combine it with a second such path in the same query; both were
    measured 3-4x slower than this exact form.

    Returns (matched_qids, succeeded). succeeded=False (query raised --
    timeout, rate-limit, etc.) MUST NOT be treated the same as "queried
    fine, zero matches": a batch of up to 50 QIDs pushed through one
    query is exactly the kind of call that can time out, and silently
    caching an empty result as a confirmed negative for every QID in
    that batch poisons the cache forever with wrong is_org/is_loc=False
    flags -- confirmed to happen for Q889 "Afghanistan" (genuinely
    is_loc=True on a fresh re-check, but a failed batch had it cached
    False, which fed a false 'structurally weak' LLM-fallback trigger
    and a real wrong-answer regression)."""
    if not qids:
        return set(), True
    values_clause = " ".join(f"wd:{q}" for q in qids)
    query = f"""
    SELECT ?item WHERE {{
      VALUES ?item {{ {values_clause} }}
      ?item wdt:P31/wdt:P279* wd:{root_qid} .
    }}
    """
    STATS["sparql_calls"] += 1
    try:
        bindings = entity_linker._sparql_query(query)
    except Exception:
        return set(), False
    return {b["item"]["value"].split("/")[-1] for b in bindings}, True


_TYPE_CHECK_CHUNK_SIZE = 15  # calibrated: ~14s/chunk reliably at this size; a
# single 50-QID batch was measured at 32s and, at least once, returned an
# incomplete match set with NO exception raised (Q889 "Afghanistan" silently
# missing from a supposedly-successful 50-QID result) -- Wikidata's endpoint
# appears to soft-truncate expensive property-path queries under load rather
# than erroring, which the try/except in _check_type_membership can't catch.
# Smaller chunks reduce how often any single query gets pushed close to
# whatever that soft limit is.


def _attach_type_flags(candidates: list) -> None:
    """Mutates each candidate dict in place, adding is_org/is_loc. Only
    ever queries Wikidata for QIDs not already in _type_cache, chunked to
    reduce large-batch timeout/truncation risk (see _TYPE_CHECK_CHUNK_SIZE).
    A QID is only ever cached from a chunk that actually succeeded -- one
    that failed is left uncached (defaults to False for THIS call only) so
    it gets retried next time, instead of being poisoned as a permanent
    false negative."""
    missing = [c["qid"] for c in candidates if c["qid"] not in _type_cache]
    newly_cached = False
    for i in range(0, len(missing), _TYPE_CHECK_CHUNK_SIZE):
        chunk = missing[i:i + _TYPE_CHECK_CHUNK_SIZE]
        org_hits, org_ok = _check_type_membership(chunk, ORG_ROOT_QID)
        loc_hits, loc_ok = _check_type_membership(chunk, LOC_ROOT_QID)
        if org_ok and loc_ok:
            for qid in chunk:
                _type_cache[qid] = {"is_org": qid in org_hits, "is_loc": qid in loc_hits}
            newly_cached = True
    if newly_cached:
        save_type_cache()
    for c in candidates:
        flags = _type_cache.get(c["qid"], {"is_org": False, "is_loc": False})
        c["is_org"] = flags["is_org"]
        c["is_loc"] = flags["is_loc"]


def get_candidates_exact(label: str, lang: str, limit: int = 15) -> list:
    key = ("exact", label, lang)
    if key in _cand_cache and _cache_entry_is_fresh(_cand_cache[key]):
        # Re-derive is_org/is_loc from the authoritative _type_cache on EVERY
        # call, even a cache hit -- _cand_cache bakes those flags into the
        # stored candidate dicts at write time, so a cache hit alone would
        # keep serving whatever value was true back when this entry was
        # written, forever, even after _type_cache has since self-healed a
        # bad one. _attach_type_flags is a cheap no-op once _type_cache
        # already has good data for these qids.
        cached = _cand_cache[key]
        _attach_type_flags(cached)
        return cached
    query = f"""
    SELECT ?item ?sitelinks ?desc ?country ?itemLabel (GROUP_CONCAT(DISTINCT ?type;separator=",") AS ?types) WHERE {{
      {{ ?item rdfs:label "{entity_linker._escape(label)}"@{lang}. }}
      UNION
      {{ ?item skos:altLabel "{entity_linker._escape(label)}"@{lang}. }}
      OPTIONAL {{ ?item wikibase:sitelinks ?sitelinks. }}
      OPTIONAL {{ ?item schema:description ?desc. FILTER(LANG(?desc) = "en") }}
      OPTIONAL {{ ?item wdt:P17 ?country. }}
      OPTIONAL {{ ?item wdt:P31 ?type. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} GROUP BY ?item ?sitelinks ?desc ?country ?itemLabel ORDER BY DESC(?sitelinks) LIMIT {limit}
    """
    STATS["sparql_calls"] += 1
    try:
        bindings = entity_linker._sparql_query(query)
    except Exception:
        bindings = []
    out, seen = [], set()
    for b in bindings:
        qid = b["item"]["value"].split("/")[-1]
        if qid in seen:
            continue
        seen.add(qid)
        types = b.get("types", {}).get("value", "")
        out.append({
            "qid": qid,
            "sitelinks": int(b["sitelinks"]["value"]) if "sitelinks" in b else 0,
            "country": b["country"]["value"].split("/")[-1] if "country" in b else None,
            "is_human": "Q5" in [t.split("/")[-1] for t in types.split(",") if t],
            "label": b["itemLabel"]["value"] if "itemLabel" in b else None,
            "description": b["desc"]["value"] if "desc" in b else None,
        })
    _attach_type_flags(out)
    _cand_cache[key] = out
    _maybe_save_cand_cache()
    return out


def get_candidates_fuzzy(label: str, lang: str, limit: int = 50) -> list:
    """wbsearchentities -> converted into the SAME candidate shape (with
    sitelinks/country/type via follow-up SPARQL) so it feeds the identical
    reranking logic. Its own top-1 is never trusted directly (validated:
    wrong for TMC/BJP).

    AUTOMATED compound-word discovery (no hardcoded aliases): widened to
    limit=50 (still the same general-purpose API call, just asking for
    more results) and results are re-sorted to prefer candidates whose
    Wikidata label is a STRICTLY LONGER extension of the search mention
    (e.g. "বাম" -> "বামফ্রন্ট") over same-length generic-word senses.
    This is what surfaces Left Front (Q1455015) for bare "বাম" -- it was
    present at raw rank 46/50, buried under generic "বাম"-labeled
    dictionary senses; the compound-extension preference brings it to the
    front automatically, for any mention with this pattern, not just this
    one word."""
    key = ("fuzzy", label, lang)
    if key in _cand_cache and _cache_entry_is_fresh(_cand_cache[key]):
        # See get_candidates_exact's cache-hit comment -- same reasoning.
        cached = _cand_cache[key]
        _attach_type_flags(cached)
        return cached
    STATS["wbsearch_calls"] += 1
    params = {
        "action": "wbsearchentities", "search": label, "language": lang,
        "uselang": lang, "format": "json", "limit": limit, "type": "item",
    }
    url = "https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "IndicClaimVer-research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            res = json.load(r)
    except Exception:
        _cand_cache[key] = []
        _maybe_save_cand_cache()
        time.sleep(1.0)
        return []
    time.sleep(1.0)  # politeness delay against wbsearchentities' own rate limit (429 confirmed earlier)
    search_hits = res.get("search", [])
    compound_qids = [h.get("id") for h in search_hits
                      if h.get("id", "").startswith("Q")
                      and h.get("label", "").startswith(label) and len(h.get("label", "")) > len(label)]
    other_qids = [h.get("id") for h in search_hits
                   if h.get("id", "").startswith("Q") and h.get("id") not in compound_qids]
    qids = compound_qids + other_qids  # compound extensions checked first

    # PERFORMANCE FIX: batch ALL qids into ONE SPARQL query via VALUES,
    # instead of one query per candidate (that was 50 sequential queries
    # x 1s politeness delay = 50+s per wbsearch call, which is what
    # stalled the first full run before it produced a single progress line).
    out = []
    if qids:
        values_clause = " ".join(f"wd:{q}" for q in qids)
        q = f"""
        SELECT ?item ?sitelinks ?country ?itemLabel ?desc (GROUP_CONCAT(DISTINCT ?type;separator=",") AS ?types) WHERE {{
          VALUES ?item {{ {values_clause} }}
          OPTIONAL {{ ?item wikibase:sitelinks ?sitelinks. }}
          OPTIONAL {{ ?item wdt:P17 ?country. }}
          OPTIONAL {{ ?item wdt:P31 ?type. }}
          OPTIONAL {{ ?item schema:description ?desc. FILTER(LANG(?desc) = "en") }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} GROUP BY ?item ?sitelinks ?country ?itemLabel ?desc
        """
        STATS["sparql_calls"] += 1
        try:
            b = entity_linker._sparql_query(q)
        except Exception:
            b = []
        by_qid = {}
        for row in b:
            qid = row["item"]["value"].split("/")[-1]
            types = row.get("types", {}).get("value", "")
            entry = {
                "qid": qid,
                "sitelinks": int(row["sitelinks"]["value"]) if "sitelinks" in row else 0,
                "country": row["country"]["value"].split("/")[-1] if "country" in row else None,
                "is_human": "Q5" in [t.split("/")[-1] for t in types.split(",") if t],
                "is_compound_match": qid in compound_qids,
                "label": row["itemLabel"]["value"] if "itemLabel" in row else None,
                "description": row["desc"]["value"] if "desc" in row else None,
            }
            # a QID can appear on multiple rows (different ?type per row via
            # OPTIONAL before GROUP_CONCAT collapses per sitelinks/country
            # combo) -- keep the one with the most info / highest sitelinks
            if qid not in by_qid or entry["sitelinks"] > by_qid[qid]["sitelinks"]:
                by_qid[qid] = entry
        out = list(by_qid.values())
    # Compound-extension candidates first (as a group), sitelinks-ranked
    # within each group -- this is what lets Left Front (fewer sitelinks
    # than the generic "বাম" senses) still win when it's the only
    # candidate that's actually a specific named entity rather than a
    # dictionary sense of the bare word.
    out.sort(key=lambda c: (not c["is_compound_match"], -c["sitelinks"]))
    _attach_type_flags(out)
    _cand_cache[key] = out
    _maybe_save_cand_cache()
    return out


def generate_candidates_v1_shortcircuit(mention: str, lang: str) -> tuple:
    """ORIGINAL (Layer-2-buggy) cascade, kept only for old-vs-new
    comparison in validation. Stops at the first non-empty stage
    regardless of candidate quality -- diagnosed root cause of the
    Modi/Mahua candidate-generation failures. Not used by resolve_document
    any more; superseded by generate_candidates() below."""
    cands = get_candidates_exact(mention, lang)
    if cands:
        return cands, "exact"
    if lang != "en" and all(ord(c) < 128 for c in mention if c.isalpha()):
        cands = get_candidates_exact(mention, "en")
        if cands:
            return cands, "code_mixed"
    if lang == "bn":
        stripped = strip_bn_suffix(mention)
        if stripped:
            cands = get_candidates_exact(stripped, lang)
            if cands:
                return cands, "morphology"
    cands = get_candidates_fuzzy(mention, lang)
    if cands:
        return cands, "wbsearch"
    return [], "none"


def _pool_is_weak(cands: list, entity_group: str) -> bool:
    """Type-aware weak-result check (Layer 2 fix). Empty is always weak.
    For PER mentions, a pool containing zero human-flagged candidates is
    weak -- this is what let Modi (15 candidates, 0 human) and Mahua (15
    candidates, 0 human) silently short-circuit before wbsearch ever ran,
    even though wbsearch demonstrably finds the correct person in both
    cases. ORG/LOC use the same idea via is_org/is_loc (P279* transitive
    closure to a single root class -- see ORG_ROOT_QID/LOC_ROOT_QID),
    rather than an enumerated subtype list. MISC/other: no type signal
    exists, so this rule does not weaken those pools."""
    if not cands:
        return True
    if entity_group == "PER":
        return not any(c["is_human"] for c in cands)
    if entity_group == "ORG":
        return not any(c.get("is_org") for c in cands)
    if entity_group == "LOC":
        return not any(c.get("is_loc") for c in cands)
    return False


def _type_filtered_pool(candidates: list, entity_group: str) -> list:
    """Shared type-aware filter used by pick_baseline/score_candidates: PER
    -> human-only, ORG -> organization-only (is_org), LOC -> geographic-
    entity-only (is_loc), each falling back to the full pool if the filter
    would empty it (a real candidate list, just none flagged -- happens
    when Wikidata doesn't expose P31 for that item). MISC/other: unchanged,
    no filter applied."""
    if entity_group == "PER":
        typed = [c for c in candidates if c.get("is_human")]
    elif entity_group == "ORG":
        typed = [c for c in candidates if c.get("is_org")]
    elif entity_group == "LOC":
        typed = [c for c in candidates if c.get("is_loc")]
    else:
        return candidates
    return typed if typed else candidates


def _dedupe_candidates(cands: list) -> list:
    merged = {}
    for c in cands:
        qid = c["qid"]
        if qid not in merged:
            merged[qid] = dict(c)
        else:
            existing = merged[qid]
            if c.get("is_compound_match") and not existing.get("is_compound_match"):
                existing["is_compound_match"] = True
            if c.get("label") and not existing.get("label"):
                existing["label"] = c["label"]
            if c.get("description") and not existing.get("description"):
                existing["description"] = c["description"]
    return list(merged.values())


def generate_candidates(mention: str, lang: str, entity_group: str = "MISC") -> tuple:
    """Phase A (Layer 2 fix applied). Returns (candidates, source_label,
    by_source) -- by_source is a dict of the RAW per-stage candidate lists
    (before merging) for trace/validation printing.

    Control-flow change from v1: a stage's non-empty result no longer
    unconditionally stops the cascade. Later stages (especially wbsearch)
    still run when the pool-so-far is "weak" (see _pool_is_weak), and
    their results are MERGED (deduped by qid) into the final pool rather
    than replacing earlier stages. Ranking (pick_baseline / semantic
    reranking) then decides from the full merged pool -- candidate quality,
    not just candidate existence, now determines whether the search
    continues. No stage removed or reordered; this only changes when the
    cascade stops and whether results accumulate or get discarded."""
    all_cands = []
    sources = []
    by_source = {}
    fuzzy_search_term = mention  # refined below if morphology stripping applies

    tqdm.write(f"      [candidates] mention={mention!r}  lang={lang}  entity_group={entity_group}")

    cands = get_candidates_exact(mention, lang)
    tqdm.write(f"        stage=exact        -> {len(cands)} candidate(s): "
               f"{[(c['qid'], c.get('label')) for c in cands[:5]]}")
    if cands:
        all_cands += cands
        sources.append("exact")
        by_source["exact"] = cands

    if _pool_is_weak(all_cands, entity_group) and lang != "en" and all(ord(c) < 128 for c in mention if c.isalpha()):
        tqdm.write(f"        pool weak ({entity_group}) and mention is Latin-script -> code-mixed retry (lang=en)")
        cc = get_candidates_exact(mention, "en")
        tqdm.write(f"        stage=code_mixed   -> {len(cc)} candidate(s): "
                   f"{[(c['qid'], c.get('label')) for c in cc[:5]]}")
        if cc:
            all_cands += cc
            sources.append("code_mixed")
            by_source["code_mixed"] = cc
            STATS["code_mixed_used_count"] += 1

    if _pool_is_weak(all_cands, entity_group) and lang == "bn":
        stripped = strip_bn_suffix(mention)
        if stripped:
            tqdm.write(f"        pool weak ({entity_group}) and lang=bn -> morphology strip: "
                       f"{mention!r} -> {stripped!r}")
            mc = get_candidates_exact(stripped, lang)
            tqdm.write(f"        stage=morphology   -> {len(mc)} candidate(s): "
                       f"{[(c['qid'], c.get('label')) for c in mc[:5]]}")
            if mc:
                all_cands += mc
                sources.append("morphology")
                by_source["morphology"] = mc
                fuzzy_search_term = stripped  # wbsearch fallback must use the
                # same refined term morphology already proved matches this
                # entity's script/spelling, not the original inflected form
                # (verified: wbsearch on the raw suffixed "মহুয়ার" finds
                # nothing at all; wbsearch on the stripped "মহুয়া" finds
                # Mahua Moitra at rank 1/37).
                STATS["morphology_used_count"] += 1
        else:
            tqdm.write(f"        pool weak ({entity_group}) and lang=bn -> "
                       f"morphology: no suffix rule matched {mention!r}")

    if _pool_is_weak(all_cands, entity_group):
        tqdm.write(f"        pool still weak ({entity_group}) -> wbsearch fuzzy fallback "
                   f"on {fuzzy_search_term!r}")
        fc = get_candidates_fuzzy(fuzzy_search_term, lang)
        tqdm.write(f"        stage=wbsearch     -> {len(fc)} candidate(s): "
                   f"{[(c['qid'], c.get('label')) for c in fc[:5]]}")
        if fc:
            all_cands += fc
            sources.append("wbsearch")
            by_source["wbsearch"] = fc
            STATS["wbsearch_used_count"] += 1

    deduped = _dedupe_candidates(all_cands)
    source_label = "+".join(sources) if sources else "none"
    tqdm.write(f"        merged pool: {len(deduped)} candidate(s)  source={source_label}")
    return deduped, source_label, by_source


# ---------------------------------------------------------------------
# Last-resort LLM fallback (NOT yet wired into generate_candidates/
# resolve_document -- standalone, isolated-tested first). Only ever
# produces a NAME, never a QID/description directly -- the caller must
# re-search Wikidata with that name, so the final answer stays grounded
# in a real Wikidata lookup, not an LLM guess.
# ---------------------------------------------------------------------
_LLM_CACHE_PATH = _shard_cache_path(Path(__file__).parent / "llm_alias_cache.json")
_llm_cache = {}


def _load_llm_cache():
    global _llm_cache
    merged_path = _merged_cache_path(_LLM_CACHE_PATH)
    if merged_path.exists():
        with open(merged_path, encoding="utf-8") as f:
            _llm_cache = json.load(f)
        print(f"[integrated_pipeline] Loaded {len(_llm_cache)} cached LLM resolutions from the shared merged cache", flush=True)
    if _LLM_CACHE_PATH.exists():
        with open(_LLM_CACHE_PATH, encoding="utf-8") as f:
            own = json.load(f)
        _llm_cache.update(own)
        print(f"[integrated_pipeline] Loaded {len(own)} cached LLM resolutions from this shard's own disk cache "
              f"({len(_llm_cache)} total after merging with the shared cache)", flush=True)


def save_llm_cache():
    with open(_LLM_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_llm_cache, f, ensure_ascii=False)


_load_llm_cache()

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


LLM_SYSTEM_PROMPT = (
    "You identify what a short mention refers to in a news context, which "
    "may be in English, Hindi, or Bengali. Reply with ONLY the entity's "
    "full name in English. If it's an abbreviation, give its English "
    "expansion. If unsure, reply exactly: NONE. No explanation."
)


def resolve_via_llm(mention: str, context_sentence: str, entity_group: str) -> tuple:
    """Returns (expanded_name_or_None, was_cache_hit). One call, capped
    output tokens, disk-cached by (mention, entity_group, context) so a
    mention is never re-billed on a second run."""
    key = f"{mention}||{entity_group}||{context_sentence}"
    if key in _llm_cache:
        return _llm_cache[key], True

    if not config.OPENAI_API_KEY:
        tqdm.write("[integrated_pipeline] OPENAI_API_KEY not set -- skipping LLM fallback.")
        return None, False

    client = _get_openai_client()
    user_msg = f"Mention: {mention} ({entity_group})\nContext: {context_sentence}"
    try:
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=30,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip()
        tqdm.write(f"[integrated_pipeline] LLM call for {mention!r} -> {answer!r} "
                   f"(usage: {resp.usage.prompt_tokens} prompt + {resp.usage.completion_tokens} completion tokens)")
    except Exception as e:
        tqdm.write(f"[integrated_pipeline] LLM call failed for {mention!r}: {e}")
        answer = None

    if answer and answer.strip().upper() != "NONE" and answer.strip().lower() != mention.strip().lower():
        result = answer.strip()
    else:
        result = None

    _llm_cache[key] = result
    save_llm_cache()
    return result, False


def _candidate_from_qid(qid: str) -> dict | None:
    """Fetches the same fields get_candidates_exact() returns, for a QID
    already known (read off a Wikipedia sitelink) rather than searched by
    label. Used only by resolve_via_wikipedia() below."""
    query = f"""
    SELECT ?sitelinks ?desc ?country ?itemLabel (GROUP_CONCAT(DISTINCT ?type;separator=",") AS ?types) WHERE {{
      wd:{qid} rdfs:label ?itemLabel. FILTER(LANG(?itemLabel) = "en")
      OPTIONAL {{ wd:{qid} wikibase:sitelinks ?sitelinks. }}
      OPTIONAL {{ wd:{qid} schema:description ?desc. FILTER(LANG(?desc) = "en") }}
      OPTIONAL {{ wd:{qid} wdt:P17 ?country. }}
      OPTIONAL {{ wd:{qid} wdt:P31 ?type. }}
    }} GROUP BY ?sitelinks ?desc ?country ?itemLabel LIMIT 1
    """
    STATS["sparql_calls"] += 1
    try:
        bindings = entity_linker._sparql_query(query)
    except Exception:
        return None
    if not bindings:
        return None
    b = bindings[0]
    types = b.get("types", {}).get("value", "")
    return {
        "qid": qid,
        "sitelinks": int(b["sitelinks"]["value"]) if "sitelinks" in b else 0,
        "country": b["country"]["value"].split("/")[-1] if "country" in b else None,
        "is_human": "Q5" in [t.split("/")[-1] for t in types.split(",") if t],
        "label": b["itemLabel"]["value"] if "itemLabel" in b else None,
        "description": b["desc"]["value"] if "desc" in b else None,
    }


def resolve_via_wikipedia(name: str) -> dict | None:
    """Last-resort grounding step, tried only when Wikidata exact AND fuzzy
    search on an LLM-expanded name both come back empty. Reuses Wikipedia's
    own redirect graph as a free, community-maintained alias dictionary --
    Wikipedia often has an exact page or redirect for a name Wikidata's own
    alias/label data never captured. Reproduced on a real case: "Delhi
    Election Commission" has zero Wikidata presence under exact or fuzzy
    search, but Wikipedia has an exact redirect from that title to "Delhi
    State Election Commission", QID Q110130785 -- the entity was always
    real, the problem was canonical naming, not coverage.

    ONLY trusts an EXACT title/redirect match -- MediaWiki's `redirects=1`
    follows an existing redirect page verbatim, it does not do fuzzy/
    best-guess search. A landing on a disambiguation page is treated as
    not-found, same as this pipeline never trusts wbsearch's own fuzzy
    top-1 directly -- this is the same trust discipline already used
    everywhere else in the cascade, just applied to a new source.

    Wikipedia is used ONLY to resolve a name to a QID. Once found, this
    function reads that QID's data from WIKIDATA (via _candidate_from_qid)
    and returns -- nothing from Wikipedia's own article content ever
    enters the candidate pool. The graph still only ever contains
    authentic Wikidata QIDs; Wikidata still defines every fact used
    downstream. Disk-cached (by name) like every other network call here,
    and never caches a network failure as a negative -- retried next run,
    not poisoned (see the type-cache-poisoning bug this exact discipline
    was already fixed for elsewhere in this file).

    Returns a candidate dict in the same shape as get_candidates_exact()
    (so it merges into the same pool/reranking untouched), or None."""
    key = ("wikipedia", name)
    if key in _cand_cache and _cache_entry_is_fresh(_cand_cache[key]):
        cached = _cand_cache[key]
        if cached:
            _attach_type_flags(cached)
        return cached[0] if cached else None

    params = {
        "action": "query", "titles": name, "redirects": "1",
        "prop": "pageprops", "ppprop": "wikibase_item|disambiguation",
        "format": "json",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "IndicClaimVer-research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            res = json.load(r)
    except Exception as e:
        tqdm.write(f"[integrated_pipeline] Wikipedia resolver call failed for {name!r}: {e}")
        return None  # network failure -- never cached, retried next call

    pages = res.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)
    if not page or "missing" in page:
        _cand_cache[key] = []
        _maybe_save_cand_cache()
        return None
    pageprops = page.get("pageprops", {})
    if "disambiguation" in pageprops:
        _cand_cache[key] = []
        _maybe_save_cand_cache()
        return None  # landed on a disambiguation page -- not a real match
    qid = pageprops.get("wikibase_item")
    if not qid:
        _cand_cache[key] = []
        _maybe_save_cand_cache()
        return None

    cand = _candidate_from_qid(qid)
    if cand is None:
        _cand_cache[key] = []
        _maybe_save_cand_cache()
        return None
    _cand_cache[key] = [cand]
    _maybe_save_cand_cache()
    _attach_type_flags([cand])
    return cand


LLM_SYSTEM_PROMPT_TIER2 = (
    "You identify what a short mention refers to in a news context, which "
    "may be in English, Hindi, or Bengali. You are given the claim it "
    "appeared in, plus related evidence passages that may contain more "
    "context about the same story. Reply with ONLY the entity's full name "
    "in English. If still unsure, reply exactly: NONE. No explanation."
)


def get_related_evidence_context(claim_text: str, pool_id_to_text: dict, top_k: int = 3) -> list:
    """Top-k evidence TEXTS (not IDs) by embedding cosine similarity to
    claim_text. Reuses embeddings.encode()/dense_retrieve_vec() -- pure
    numpy, no Neo4j round-trip. Requires embeddings.prepare_pool_embeddings()
    to have already been called by the caller (populates the module-level
    pool index dense_retrieve_vec reads from); returns [] if it hasn't, or
    if none of the retrieved IDs are in pool_id_to_text."""
    from subtask1 import embeddings as _emb
    if _emb._pool_ids is None:
        return []
    claim_vec = _emb.encode(claim_text, is_query=True)
    top_ids = _emb.dense_retrieve_vec(claim_vec, top_k=top_k)
    return [pool_id_to_text[eid] for eid in top_ids if eid in pool_id_to_text]


def resolve_via_llm_tier2(mention: str, claim_text: str, related_evidence: list, entity_group: str) -> tuple:
    """Second-tier LLM fallback -- only ever called when resolve_via_llm()
    (tier 1: mention + local sentence only) already returned None. Adds
    related evidence (top-k by embedding similarity to the CLAIM, not the
    gold label -- works the same way for a genuinely new claim at real
    inference time) as extra context, on the theory that the disambiguating
    detail a short claim headline lacks often exists in the fuller story
    the evidence pool covers. Same trust model as tier 1: only ever
    produces a name, re-grounded via the same Wikidata search functions,
    never a QID/description directly. Own cache namespace (key suffixed
    __t2) so it can never collide with a tier-1 cached None for the same
    mention -- the two questions ("just the mention+sentence" vs "mention+
    claim+related evidence") are genuinely different prompts."""
    key = f"{mention}||{entity_group}||{claim_text}||__t2"
    if key in _llm_cache:
        return _llm_cache[key], True

    if not config.OPENAI_API_KEY or not related_evidence:
        return None, False

    client = _get_openai_client()
    evidence_block = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(related_evidence))
    user_msg = f"Mention: {mention} ({entity_group})\nClaim: {claim_text}\n\nRelated evidence:\n{evidence_block}"
    try:
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT_TIER2},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=30,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip()
        tqdm.write(f"[integrated_pipeline] LLM tier-2 call for {mention!r} -> {answer!r} "
                   f"(usage: {resp.usage.prompt_tokens} prompt + {resp.usage.completion_tokens} completion tokens)")
    except Exception as e:
        tqdm.write(f"[integrated_pipeline] LLM tier-2 call failed for {mention!r}: {e}")
        answer = None

    if answer and answer.strip().upper() != "NONE" and answer.strip().lower() != mention.strip().lower():
        result = answer.strip()
    else:
        result = None

    _llm_cache[key] = result
    save_llm_cache()
    return result, False


def pick_baseline_v1_hardfilter(candidates: list, entity_group: str) -> str | None:
    """ORIGINAL (Layer-3-buggy) baseline pick, kept only for old-vs-new
    comparison. "if compound: pool = compound" is an absolute filter --
    it excludes every non-compound candidate even when one is objectively
    better, which is what made Ravinder Raina (non-compound, sitelinks=4,
    correct) permanently unreachable behind 10 unrelated same-prefix
    "Rabindra *" institutions. Not used by resolve_document any more."""
    if not candidates:
        return None
    if entity_group == "PER":
        humans = [c for c in candidates if c["is_human"]]
        pool = humans if humans else candidates
    else:
        pool = candidates
    compound = [c for c in pool if c.get("is_compound_match")]
    if compound:
        pool = compound
    return max(pool, key=lambda c: c["sitelinks"])["qid"]


def pick_baseline(candidates: list, entity_group: str) -> str | None:
    """Phase B1 (Layer 3 fix applied): type-aware filter, then a SOFT
    compound-match preference instead of an absolute one. Compound-match
    candidates are preferred, but only when they are actually competitive
    -- a non-compound candidate with strictly higher sitelinks than the
    best compound candidate is no longer discarded outright. This does
    not change existing behavior for cases where compound genuinely wins
    (e.g. Left Front vs. generic "left" direction), and it does not, by
    itself, guarantee the objectively-correct candidate always wins where
    the wrong candidate is simply more notable (e.g. Ravinder Raina,
    sitelinks=4, still loses to Rabindra Nath Chaudhuri, sitelinks=6, under
    sitelinks-only ranking) -- that class of failure needs a context
    signal beyond sitelinks, which is exactly what Phase 5 semantic
    reranking would add. This fix's guarantee is narrower and still real:
    a globally better candidate is never made structurally unreachable
    just for lacking a literal string-prefix match."""
    if not candidates:
        return None
    pool = _type_filtered_pool(candidates, entity_group)
    compound = [c for c in pool if c.get("is_compound_match")]
    non_compound = [c for c in pool if not c.get("is_compound_match")]
    if compound and non_compound:
        best_compound = max(compound, key=lambda c: c["sitelinks"])
        best_non_compound = max(non_compound, key=lambda c: c["sitelinks"])
        winner = best_compound if best_compound["sitelinks"] >= best_non_compound["sitelinks"] else best_non_compound
        return winner["qid"]
    if compound:
        return max(compound, key=lambda c: c["sitelinks"])["qid"]
    return max(pool, key=lambda c: c["sitelinks"])["qid"]


# ---------------------------------------------------------------------
# Semantic reranking: context-sentence-vs-Wikidata-description matching.
# Runs AFTER candidate generation and lexical ranking (pick_baseline),
# over the same candidate pool -- reranks, never generates new candidates.
# Moved here (from the former standalone experiments/semantic_reranking.py)
# so resolve_document() is the single entry point production callers use;
# semantic_reranking.py now re-imports these for backward compatibility.
# ---------------------------------------------------------------------
SENTENCE_ENDERS = {
    "en": r"[.?!]",
    "hi": r"।",
    "bn": r"[।৷]",
}


def split_sentences(text: str, lang: str) -> list:
    pattern = SENTENCE_ENDERS.get(lang, SENTENCE_ENDERS["en"])
    parts = re.split(f"({pattern})", text)
    sentences, buf = [], ""
    for part in parts:
        buf += part
        if re.fullmatch(pattern, part):
            sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return [s for s in sentences if s]


def get_context_sentence(text: str, mention: str, lang: str) -> str:
    """The local sentence containing the mention -- never the whole
    article, never the bare mention alone. Falls back to the full text
    only when there's just one sentence, or the mention spans a boundary."""
    sentences = split_sentences(text, lang)
    if len(sentences) <= 1:
        return text.strip()
    for sent in sentences:
        if mention in sent:
            return sent
    return text.strip()


_EMBED_CACHE_PATH = _shard_cache_path(Path(__file__).parent / "semantic_embed_cache.pkl")
_embed_cache = {}
_embed_model = None


def _load_embed_cache():
    global _embed_cache
    if _EMBED_CACHE_PATH.exists():
        with open(_EMBED_CACHE_PATH, "rb") as f:
            _embed_cache = pickle.load(f)
        print(f"[integrated_pipeline] Loaded {len(_embed_cache)} cached embeddings from disk", flush=True)


def save_embed_cache():
    with open(_EMBED_CACHE_PATH, "wb") as f:
        pickle.dump(_embed_cache, f)


_load_embed_cache()


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from subtask1 import embeddings as _emb
        _embed_model = _emb._get_model()  # reuse the already-loaded E5 model instance
    return _embed_model


def _embed_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed(text: str, is_query: bool) -> tuple:
    """Returns (vector: np.ndarray, was_cache_hit: bool). Own cache file
    (NOT config.EMBED_CACHE_PATH) -- candidate-description embeddings are
    a different kind of text than the claim/evidence retrieval pool, kept
    separate to avoid mixing the two caches."""
    prefixed = f"{'query' if is_query else 'passage'}: {text}"
    h = _embed_hash(prefixed)
    if h in _embed_cache:
        return np.asarray(_embed_cache[h], dtype=np.float32), True
    vec = _get_embed_model().encode(prefixed, convert_to_numpy=True).astype(np.float32)
    _embed_cache[h] = vec.tolist()
    return vec, False


def get_description_embedding(description: str) -> tuple:
    """Embed a CANDIDATE DESCRIPTION only (never the label). passage-side
    of the asymmetric E5 convention."""
    if not description:
        return None, False
    return _embed(description, is_query=False)


def get_context_embedding(context_sentence: str) -> tuple:
    """Embed the LOCAL CONTEXT SENTENCE only (never the mention alone,
    never the full article). query-side of the asymmetric E5 convention."""
    return _embed(context_sentence, is_query=True)


def cosine_sim(a, b) -> float:
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


# SCORING FORMULA:
#   rank_score(c)     = c.sitelinks / max(1, max_sitelinks_in_pool)   [0..1]
#   compound_bonus(c) = COMPOUND_BONUS if c.is_compound_match else 0
#   semantic_score(c) = min-max normalized cosine(context_embedding,
#                        description_embedding(c)) WITHIN this mention's
#                        own pool -- raw E5 cosine similarities cluster
#                        tightly (~0.75-0.78) across an entire pool (a
#                        known E5 property), so within-pool normalization
#                        is what makes the signal discriminative.
#   FINAL_SCORE(c) = rank_score(c) + compound_bonus(c) + SEMANTIC_WEIGHT * semantic_score(c)
#   winner = argmax(FINAL_SCORE) -- this call REPLACES pick_baseline's
#   plain sitelinks/compound comparison when semantic reranking is
#   enabled, since rank_score + compound_bonus already reproduce
#   pick_baseline's preference direction; semantic_score is the added
#   signal on top of it.
COMPOUND_BONUS = 0.05
SEMANTIC_WEIGHT = 1.0

# TUNABLE -- calibrated from data_test/scripts/semantic_full_validation.py's
# known-case set (top-1 vs top-2 final_score margin, free/no-LLM run):
#   known-WRONG (Layer-1 alias-coverage gap, correct entity never in the
#   pool at all): MCA margin=0.132, DEC margin=0.0003
#   known-CORRECT: Mahua margin=0.195, Ravindra Jadeja margin=0.686,
#   NDA margin=1.055, Modi margin=1.431
# Clean separation in the gap between 0.132 and 0.195 -- 0.15 sits roughly
# centered in it. Below this margin, resolve_document's LLM fallback gets
# a chance even when the pool isn't structurally weak (i.e. it already
# contains a candidate of the right type, just possibly the wrong specific
# entity, which is exactly the MCA/DEC failure mode _pool_is_weak cannot
# see). Single-candidate pools (margin=None, nothing to be unconfident
# against) never trigger this.
#
# KNOWN LIMITATION (confirmed on MCA): the trigger correctly fires and
# resolve_via_llm() correctly retrieves the right entity into the pool,
# but it can still lose the RE-SCORED ranking to a more "famous" wrong
# candidate -- rank_score (sitelinks-based notability) can outweigh a
# strong semantic_score when the correct entity is a low-profile Wikidata
# item. Measured: Ministry of Corporate Affairs (sitelinks=7, semantic=
# 0.810) still loses to Malaysian Chinese Association (sitelinks=19,
# semantic=0.778) -- rank_score's gap (0.364) swamps semantic_score's
# advantage (0.032). Not fixed here -- would need either reweighting
# rank_score/SEMANTIC_WEIGHT (risks regressing the already-validated
# Modi/Mahua/Ravindra/NDA cases) or a targeted llm_alias-sourced bonus
# (same pattern as COMPOUND_BONUS, scoped to only affect candidates the
# LLM stage itself retrieved). Left as a documented open problem.
CONFIDENCE_MARGIN_THRESHOLD = 0.15


def score_candidates(candidates: list, entity_group: str, context_vec) -> list:
    """Returns candidates (type-filtered same as pick_baseline via
    _type_filtered_pool) each carrying every component score plus
    final_score, sorted descending. Does not mutate `candidates`."""
    pool = _type_filtered_pool(candidates, entity_group)
    if not pool:
        return []

    max_sitelinks = max((c.get("sitelinks", 0) for c in pool), default=0) or 1

    raw_sims = []
    for c in pool:
        desc = c.get("description")
        if desc:
            dvec, _hit = get_description_embedding(desc)
            sim = cosine_sim(context_vec, dvec) if dvec is not None else 0.0
        else:
            sim = 0.0
        raw_sims.append(sim)
    min_sim, max_sim = min(raw_sims), max(raw_sims)
    sim_range = (max_sim - min_sim) or 1e-8

    scored = []
    for c, raw_sim in zip(pool, raw_sims):
        rank_score = c.get("sitelinks", 0) / max_sitelinks
        compound_bonus = COMPOUND_BONUS if c.get("is_compound_match") else 0.0
        semantic_score = (raw_sim - min_sim) / sim_range
        final_score = rank_score + compound_bonus + SEMANTIC_WEIGHT * semantic_score
        scored.append({
            "qid": c["qid"], "label": c.get("label"), "description": c.get("description"),
            "sitelinks": c.get("sitelinks", 0), "is_compound_match": bool(c.get("is_compound_match")),
            "is_human": c.get("is_human", False),
            "rank_score": rank_score, "compound_bonus": compound_bonus,
            "raw_semantic_sim": raw_sim, "semantic_score": semantic_score,
            "final_score": final_score,
        })
    scored.sort(key=lambda s: s["final_score"], reverse=True)
    return scored


# ---------------------------------------------------------------------
# Full per-document resolution
# ---------------------------------------------------------------------
def resolve_document(mentions_with_meta: list, lang: str, full_text: str = None,
                       enable_semantic_reranking: bool = True,
                       enable_llm_fallback: bool = False,
                       enable_llm_tier2: bool = False,
                       pool_id_to_text: dict = None) -> dict:
    """mentions_with_meta: list of (mention_text, entity_group). full_text:
    the Claim/Evidence's own source text (required for semantic reranking
    -- used to extract each mention's local context sentence; pass None
    only for lexical-only use, e.g. quick smoke tests).
    Returns {mention_text: {"qid":..., "label":..., "source":..., "n_candidates":...,
    "is_human":..., "trace":[...] (only when semantic reranking ran), ...}}

    enable_semantic_reranking: default True -- Phase B1.5. Runs AFTER
    candidate generation (Phase A) and lexical ranking, over the SAME
    candidate pool; never generates new candidates. Falls back to plain
    pick_baseline() (lexical-only) if False or if full_text is None.

    enable_llm_fallback: default False -- last-resort stage, fires when
    EITHER the free cascade leaves the pool structurally weak (zero
    candidates of the expected type -- _pool_is_weak) OR, if semantic
    reranking is on, the top-1 candidate doesn't clearly beat the runner-up
    (margin < CONFIDENCE_MARGIN_THRESHOLD -- catches the case where a
    candidate of the right type exists but is the wrong specific entity,
    e.g. MCA/DEC, which structural weakness alone can't see). Uses
    resolve_via_llm() (OpenAI, paid) to get an English expansion/
    identification of the mention from context, then re-searches Wikidata
    (exact, then wbsearch) with that name -- the LLM only ever supplies a
    search string, never a QID/description directly, so the final answer
    is always grounded in a real Wikidata lookup. Defaults OFF (unlike the
    other flags) because, unlike everything else in the cascade, each call
    costs real money -- must be explicitly opted into per run, not
    silently enabled.

    enable_llm_tier2: default False -- second-stage escalation, only tried
    when enable_llm_fallback is on AND tier 1 (resolve_via_llm, mention +
    local sentence only) already returned None. Retrieves the top-3
    evidence pool texts by embedding similarity to `full_text` (via
    get_related_evidence_context()) and gives the LLM a second attempt
    with that richer context -- catches cases where the disambiguating
    detail exists in the broader story but not in the one sentence the
    mention appears in (e.g. a bare first name that's only spelled out
    with a surname in a topically related evidence passage). Requires
    pool_id_to_text and embeddings.prepare_pool_embeddings() to have
    already been called by the caller; silently does nothing without
    them (same fail-open behavior as the rest of the cascade). Scoped to
    Claim-side resolution only by convention of the caller -- using an
    Evidence item's OWN gold claim as context would be circular/leaky in
    a way using a Claim's semantically-nearest evidence isn't (see design
    discussion: this works identically for a genuinely new claim at real
    inference time, since it only needs the claim's own text plus the
    already-built evidence embedding index, not a known gold label).

    Exact match / code-mixed / morphology / wbsearch / coreference are
    unaffected by any of these flags."""
    results = {}
    raw_candidates = {}

    # Phase A + B1 (+ B1.5 semantic) for every mention
    for mention, egroup in mentions_with_meta:
        tqdm.write(f"    [resolve] mention={mention!r}  entity_group={egroup}  lang={lang}")
        cands, source, by_source = generate_candidates(mention, lang, egroup)
        ctx_sentence = get_context_sentence(full_text, mention, lang) if full_text is not None else None

        use_semantic = enable_semantic_reranking and full_text is not None and cands
        scored = None
        if use_semantic:
            ctx_vec, _hit = get_context_embedding(ctx_sentence)
            scored = score_candidates(cands, egroup, ctx_vec)

        # Last-resort LLM fallback: fires on either of two independent
        # weak-result signals. (1) structurally_weak (_pool_is_weak): the
        # pool has zero candidates of the expected type at all (the
        # Modi/Mahua-class failure). (2) confidence_weak: semantic
        # reranking's top-1 doesn't clearly beat the runner-up -- the
        # MCA/DEC-class failure, where a candidate OF the right type wins
        # but is the wrong specific entity, so structurally_weak alone
        # never fires (see CONFIDENCE_MARGIN_THRESHOLD for the calibration
        # data). Never trusted for a QID directly -- only ever produces a
        # name, re-searched through the same exact-match/wbsearch
        # functions already used everywhere else.
        structurally_weak = _pool_is_weak(cands, egroup)
        confidence_weak = bool(scored) and len(scored) >= 2 and (
            scored[0]["final_score"] - scored[1]["final_score"] < CONFIDENCE_MARGIN_THRESHOLD)
        trusted_qid = None
        if enable_llm_fallback and full_text is not None and (structurally_weak or confidence_weak):
            expanded, _llm_hit = resolve_via_llm(mention, ctx_sentence, egroup)
            if expanded is None and enable_llm_tier2 and pool_id_to_text:
                related = get_related_evidence_context(full_text, pool_id_to_text, top_k=3)
                if related:
                    STATS["llm_tier2_attempted_count"] += 1
                    expanded, _llm_hit = resolve_via_llm_tier2(mention, full_text, related, egroup)
                    if expanded:
                        STATS["llm_tier2_success_count"] += 1
            if expanded:
                exact_llm_cands = get_candidates_exact(expanded, "en")
                llm_cands = exact_llm_cands if exact_llm_cands else get_candidates_fuzzy(expanded, "en")
                wikipedia_cand = None
                if not llm_cands:
                    # Last resort before giving up on this expansion entirely:
                    # Wikidata's own exact-label AND fuzzy-search both missed
                    # this name, but Wikipedia's redirect graph might still
                    # have it (reproduced on a real case -- "Delhi Election
                    # Commission" has zero Wikidata presence under exact or
                    # fuzzy search, but Wikipedia has an exact redirect from
                    # that title to "Delhi State Election Commission", QID
                    # Q110130785). See resolve_via_wikipedia()'s docstring for
                    # why only an exact title/redirect match counts here, and
                    # why Wikipedia is used only to resolve a name to a QID,
                    # never as a data source -- the graph still only ever
                    # contains authentic Wikidata QIDs.
                    wikipedia_cand = resolve_via_wikipedia(expanded)
                    if wikipedia_cand:
                        llm_cands = [wikipedia_cand]
                        STATS["wikipedia_resolver_used_count"] += 1
                if llm_cands:
                    # TRUST-DIRECTLY check, evaluated against the pool as it
                    # stood BEFORE merging: an exact-match search on the
                    # LLM's specific expansion is a much stronger, more
                    # specific grounding signal than sitelinks-based ranking
                    # over the noisy bare-mention pool (confirmed on MCA:
                    # Ministry of Corporate Affairs, sitelinks=7, lost to
                    # Malaysian Chinese Association, sitelinks=19, by 0.33
                    # final_score even after being merged in and rescored).
                    # Guard: the candidate's raw context similarity must be
                    # no worse than the pool's own worst member -- not a
                    # fixed absolute cosine floor, since E5 similarities are
                    # known to cluster tightly (~0.75-0.78) regardless of
                    # relevance (see score_candidates' SCORING FORMULA
                    # comment), so a magic-number floor wouldn't discriminate
                    # well. Only applies when the expansion was UNAMBIGUOUS
                    # (exact match found exactly one entity, OR an exact
                    # Wikipedia title/redirect match -- same confidence tier,
                    # both are exact matches on the specific expansion, just
                    # from two different sources) -- wbsearch's own top-1 is
                    # never trusted this way, same reasoning as everywhere
                    # else it's used.
                    if use_semantic and (len(exact_llm_cands) == 1 or wikipedia_cand is not None) and scored:
                        llm_cand = exact_llm_cands[0] if exact_llm_cands else wikipedia_cand
                        desc = llm_cand.get("description")
                        if desc:
                            dvec, _hit = get_description_embedding(desc)
                            llm_raw_sim = cosine_sim(ctx_vec, dvec) if dvec is not None else 0.0
                            pool_min_sim = min(s["raw_semantic_sim"] for s in scored)
                            if llm_raw_sim >= pool_min_sim:
                                trusted_qid = llm_cand["qid"]

                    cands = _dedupe_candidates(cands + llm_cands)
                    sources_so_far = source.split("+") if source != "none" else []
                    source_tag = "wikipedia" if wikipedia_cand is not None else "llm_alias"
                    source = "+".join(sources_so_far + [source_tag])
                    by_source[source_tag] = llm_cands
                    STATS["llm_alias_used_count"] += 1
                    if trusted_qid is not None:
                        STATS["llm_alias_trusted_count"] += 1
                    # Re-derive against the ENRICHED pool -- cands may have
                    # gone from empty to non-empty, or the correct entity
                    # may now be present and win the re-scored ranking.
                    use_semantic = enable_semantic_reranking and full_text is not None and cands
                    if use_semantic:
                        ctx_vec, _hit = get_context_embedding(ctx_sentence)
                        scored = score_candidates(cands, egroup, ctx_vec)

        raw_candidates[mention] = cands

        if use_semantic:
            for s in scored:
                sqid = s["qid"]
                s["exact_score"] = 1 if sqid in {c["qid"] for c in by_source.get("exact", [])} else 0
                s["morphology_score"] = 1 if sqid in {c["qid"] for c in by_source.get("morphology", [])} else 0
                s["code_mixed_score"] = 1 if sqid in {c["qid"] for c in by_source.get("code_mixed", [])} else 0
                s["wbsearch_score"] = 1 if sqid in {c["qid"] for c in by_source.get("wbsearch", [])} else 0
                s["llm_alias_score"] = 1 if sqid in {c["qid"] for c in by_source.get("llm_alias", [])} else 0
                s["coreference_score"] = 0
            if trusted_qid is not None:
                winner = next((s for s in scored if s["qid"] == trusted_qid), scored[0] if scored else None)
            else:
                winner = scored[0] if scored else None
            qid = winner["qid"] if winner else None
            chosen_is_human = winner["is_human"] if winner else False
            chosen_label = winner["label"] if winner else None
            chosen_description = winner["description"] if winner else None
            if winner and trusted_qid is not None:
                winner_reason = (f"LLM-grounded exact match trusted directly over pool ranking "
                                  f"(final_score={winner['final_score']:.4f}, would rank "
                                  f"#{scored.index(winner) + 1}/{len(scored)} by score alone)")
            else:
                winner_reason = (f"highest FINAL_SCORE={winner['final_score']:.4f} "
                                  f"(rank={winner['rank_score']:.3f} + compound={winner['compound_bonus']:.3f} "
                                  f"+ semantic={SEMANTIC_WEIGHT}*{winner['semantic_score']:.3f})") if winner else "no candidates"
            if scored:
                tqdm.write(f"      [scoring] context={ctx_sentence[:100]!r}")
                for s in sorted(scored, key=lambda s: -s["final_score"])[:5]:
                    tqdm.write(f"        {s['qid']:12s} final={s['final_score']:.4f}  rank={s['rank_score']:.3f}  "
                               f"compound={s['compound_bonus']:.3f}  sem={s['semantic_score']:.3f}  "
                               f"sitelinks={s['sitelinks']}  | {s['label']!r} | {s['description']!r}")
        else:
            ctx_sentence = None
            scored = None
            qid = pick_baseline(cands, egroup)
            chosen_cand = next((c for c in cands if c["qid"] == qid), None)
            chosen_is_human = chosen_cand["is_human"] if chosen_cand else False
            chosen_label = chosen_cand.get("label") if chosen_cand else None
            chosen_description = chosen_cand.get("description") if chosen_cand else None
            winner_reason = "highest lexical rank (pick_baseline: sitelinks + compound-match preference)" if qid else "no candidates"

        tqdm.write(f"      [winner] qid={qid}  label={chosen_label!r}  description={chosen_description!r}  "
                   f"source={source}  reason={winner_reason}")

        results[mention] = {
            "qid": qid, "source": source, "n_candidates": len(cands),
            "is_human": chosen_is_human,
            "label": chosen_label,
            "description": chosen_description,
            "context_sentence": ctx_sentence,
            "trace": scored,
            "winner_reason": winner_reason,
            # pre_wbsearch_* records the state BEFORE any wbsearch stage
            # touches this mention, so we can later report exactly what
            # wbsearchentities changed (fixed vs. broke), not just the
            # final answer.
            "pre_wbsearch_qid": qid, "pre_wbsearch_source": source,
        }
        STATS["resolution_source"][source] += 1

    # Phase C: same-document PER coreference. Unconditionally checks every
    # PER mention against any longer same-document PER mention that is a
    # literal text-prefix match -- NOT gated on whether the short mention
    # already resolved to "a human", confidently or otherwise. Two earlier,
    # narrower gates were tried and both broke on real data before this:
    # (1) "only rescue if unresolved or non-human" missed the Rekha
    #     Sharma-class case ("রেখা শর্মা" resolves to a REAL human, just
    #     the wrong one -- a Nepali MP homonym -- while the longer "রেখা
    #     শর্মাকে" in the same document correctly resolves to the actual
    #     NCW chairperson: already-human is not evidence of already-right).
    # (2) adding a CONFIDENCE_MARGIN_THRESHOLD check on top of that also
    #     failed -- "রবিন্দ্র" wrongly resolves to cricketer Ravindra
    #     Jadeja with final_score margin 0.686 (nowhere near "weak"),
    #     simply because Jadeja dominates whatever else shares the bare
    #     name "Ravindra" in Wikidata, independent of whether he's the
    #     right person for this context. A margin computed within one
    #     mention's own candidate pool cannot detect "the correct answer
    #     isn't even a candidate in this pool" -- which is exactly what's
    #     happening here (Ravinder Raina only turns up under the longer,
    #     more specific search term "রবিন্দ্র রৈনা").
    # Checked directly against a real counter-example before removing the
    # gate rather than assuming it's safe: "Ram" vs "Ram Rajya" (a
    # governance-concept phrase, not the same referent as "Ram") -- but
    # "Ram Rajya" itself resolves with is_human=False (a 1948 film, in
    # fact), so the EXISTING is_human check on the candidate (m2, below)
    # already excludes it as a valid coreference source. That check was
    # already sufficient; the extra gates on m were solving a problem
    # that didn't need solving and missing the ones that did.
    per_mentions = [(m, eg) for m, eg in mentions_with_meta if eg == "PER"]
    for m, eg in per_mentions:
        for m2, eg2 in per_mentions:
            if m2 == m or eg2 != "PER":
                continue
            if (m2.startswith(m) and len(m2) > len(m) and results[m2]["qid"]
                    and results[m2]["is_human"] and results[m]["qid"] != results[m2]["qid"]):
                results[m]["qid"] = results[m2]["qid"]
                results[m]["label"] = results[m2]["label"]
                results[m]["description"] = results[m2]["description"]
                results[m]["source"] = "coreference"
                results[m]["is_human"] = True
                results[m]["winner_reason"] = f"coreference: adopted from same-document mention {m2!r}"
                if results[m]["trace"]:
                    for s in results[m]["trace"]:
                        s["coreference_score"] = 1 if s["qid"] == results[m2]["qid"] else 0
                STATS["coreference_used_count"] += 1
                break

    return results


if __name__ == "__main__":
    print("Module loaded. Run integrated_pipeline_run.py to execute against test1.")
