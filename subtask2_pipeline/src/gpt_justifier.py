"""
Generates a real, grounded natural-language justification for an
ALREADY-DECIDED MuRIL prediction, using GPT-4o-mini.

This does NOT reclassify -- the SUPPORTS/REFUTES label is fixed input,
never reconsidered or overridden here. GPT-4o-mini's only job is to
explain, in evidence-grounded prose, why the given label follows from
the given evidence. This replaces the plain template sentence
run_classifier_pipeline.py previously wrote for MuRIL's Justification
field ("Based on the retrieved evidence, the classifier predicts this
claim {label} with a confidence score of {conf}.") with a genuine,
evidence-specific explanation.

Disk-cached by (claim text, evidence text, prediction) so a re-run
never re-bills an already-justified record.

THREAD SAFETY: callers (regenerate_classifier_justifications.py) run
this from a ThreadPoolExecutor, many claims at once. All access to the
shared in-memory cache dict, and every write of it to disk, is guarded
by _cache_lock -- an earlier version had no lock at all, which crashed
mid-run with "dictionary changed size during iteration" (one thread's
json.dump(_cache, ...) iterating the dict while another thread's
cache[key] = text mutated it concurrently), and being a plain
open(path, "w") + json.dump with no atomicity, that crash risked
leaving a truncated/corrupted cache file on disk. _save_cache() now
writes to a temp file and atomically replaces the real one, so a crash
mid-write can never corrupt the cache -- worst case, that one save is
lost and gets redone next call, not a corrupted file every future call
fails to load.
"""
import hashlib
import json
import os
import threading

from . import config

SYSTEM_PROMPT = (
    "You are a fact-checking assistant. You are given a CLAIM, an EVIDENCE "
    "passage, and a VERDICT that has ALREADY been determined by a separate "
    "classifier -- SUPPORTS or REFUTES. Your ONLY job is to write a short, "
    "evidence-grounded justification explaining why the evidence leads to "
    "this verdict. Do NOT question, re-evaluate, or change the verdict -- "
    "treat it as fixed and correct, and justify it. Reason ONLY from the "
    "evidence text given; do not use outside knowledge or invent facts not "
    "present in it. Do not hedge (\"insufficient evidence\", \"cannot "
    "verify\", etc.) -- the verdict is already decided, your job is only "
    "to explain it. Write 2-4 sentences, roughly 60-100 words, in English."
)

_CACHE_PATH = config.OUTPUT_DIR / "gpt_justification_cache.json"
_cache = None
_client = None
_cache_lock = threading.Lock()


def _load_cache() -> dict:
    global _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        if _CACHE_PATH.exists():
            with open(_CACHE_PATH, encoding="utf-8") as f:
                _cache = json.load(f)
        else:
            _cache = {}
        return _cache


def _save_cache_locked() -> None:
    """Caller MUST already hold _cache_lock. Writes to a temp file and
    atomically replaces the real cache file, so a crash mid-write (or a
    concurrent reader) can never see/leave a truncated JSON file.

    RETRY: os.replace() on Windows can transiently fail with
    "PermissionError: [WinError 5] Access is denied" if something else
    (antivirus, search indexer, a lingering handle) has the target file
    open for a moment -- observed directly, crashing a real run at
    record ~50/174. This is not a logic bug (the lock already
    serializes which thread calls this), it's an OS-level race outside
    our control, so it's handled the same way the rest of this codebase
    handles transient external failures: retry with a short backoff
    instead of crashing the whole batch over one save."""
    import time

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _CACHE_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False)

    last_error = None
    for attempt in range(5):
        try:
            os.replace(tmp_path, _CACHE_PATH)
            return
        except PermissionError as e:
            last_error = e
            time.sleep(0.2 * (attempt + 1))
    raise last_error


def _cache_key(claim_text: str, evidence_text: str, prediction: str) -> str:
    return hashlib.sha256(
        f"{claim_text}||{evidence_text}||{prediction}".encode("utf-8")
    ).hexdigest()


def _get_client():
    global _client
    with _cache_lock:
        if _client is None:
            from openai import OpenAI
            if not config.OPENAI_API_KEY:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Add it to subtask2_pipeline/.env before running this."
                )
            _client = OpenAI(api_key=config.OPENAI_API_KEY)
        return _client


def justify(claim_text: str, evidence_text: str, prediction: str, model: str = None) -> str:
    """Returns a GPT-4o-mini-generated justification for the
    ALREADY-DECIDED `prediction` (SUPPORTS or REFUTES), grounded in
    `evidence_text`. Cached by (claim_text, evidence_text, prediction)
    -- a re-run with the exact same inputs never re-bills."""
    cache = _load_cache()
    key = _cache_key(claim_text, evidence_text, prediction)
    with _cache_lock:
        if key in cache:
            return cache[key]

    client = _get_client()
    model = model or config.OPENAI_MODEL_VERIFICATION or "gpt-4o-mini"
    user_msg = (
        f"CLAIM: {claim_text}\n\n"
        f"EVIDENCE: {evidence_text}\n\n"
        f"VERDICT (already determined, do not change): {prediction}\n\n"
        f"Write the justification."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        # Last-resort fallback if the API genuinely returns nothing usable
        # -- never leave the field empty.
        text = f"Based on the retrieved evidence, the classifier predicts this claim {prediction}."

    with _cache_lock:
        cache[key] = text
        _save_cache_locked()
    return text
