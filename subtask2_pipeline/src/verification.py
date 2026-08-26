"""
STEPS 10-12: LLM Evidence Summary + Prediction + Justification.

Single responsibility: given a claim and its top-N fusion-ranked
evidence items, fetch each item's full article text (via
article_fetcher.py), feed claim + evidence to an LLM under strict
grounding instructions, and produce exactly the FIRE 2026 Subtask-2
submission record:

    {"ID": ..., "Evidence": ..., "Prediction": "SUPPORTS"/"REFUTES", "Justification": ...}

CHANGED AGAIN (2026-07): `google.generativeai` is a fully deprecated
SDK (see https://github.com/google-gemini/deprecated-generative-ai-python)
and its model listing no longer serves newer API keys -- this caused a
404 on "gemini-2.5-flash is no longer available to new users". Migrated
to Google's current, actively-maintained `google-genai` package
(`from google import genai`, `client.models.generate_content(...)`),
using config.GEMINI_MODEL = "gemini-flash-latest" (a rolling alias, see
config.py) instead of a specific dated model string. Also explicitly
disables Gemini's "thinking" step (see _call_llm_once's docstring) to
avoid a separate, widely-reported failure mode where internal reasoning
silently consumes the entire output-token budget and returns empty
text. Every prompt, every validation rule, every retry/robustness
behavior, and both public function signatures (verify_claim,
verify_claim_folder) remain byte-identical throughout all of these
LLM-backend changes.

(Historical note, 2026-07-21): the verification-stage LLM was
originally switched from Gemini 2.5 Flash to OpenAI GPT-4o mini (the
`openai` SDK, `from openai import OpenAI`) to work around Gemini
free-tier quota limits when processing all 1009 claims.

GROUNDING DISCIPLINE: the prompt explicitly forbids the model from
using background knowledge or inventing facts not present in the
supplied evidence text -- it must reason ONLY over what was retrieved
and fetched. FIRE 2026 has exactly two labels, no "not enough
evidence" option, so the model must still commit to one of the two on
every claim -- but (per FIRE's evaluation philosophy) that commitment
must come from a genuine reading of what the evidence says, NEVER from
a mechanical "evidence is thin/missing -> REFUTES" shortcut. REFUTES is
correct only when the evidence, on inspection, actually contradicts the
claim or fails to substantiate it -- not merely because little or no
evidence was retrieved. The Justification is additionally required to
explicitly reference the retrieved evidence (e.g. "the evidence" /
"Source 1"), so the grounding is visible in the output itself, not just
enforced by instruction. Hedging phrases ("insufficient evidence",
"cannot verify", etc.) are explicitly banned in the prompt AND checked
for after generation (see BANNED_HEDGE_PHRASES / _find_hedge_phrase) --
if one slips through anyway, that's treated as a failed attempt and
retried, same as truncation or malformed JSON.

PREDICTION VALIDATION: "SUPPORTS"/"REFUTES" is checked EXACTLY
(case-sensitive) against config.VALID_PREDICTIONS.

ROBUSTNESS: the model's JSON mode structurally guarantees syntactically
valid JSON UNLESS the response is cut off by hitting the token limit
mid-string -- which is exactly what caused the original
"Unterminated string" crash (Bengali/Hindi text tokenizes far less
efficiently than English, so a token budget that's fine for an
English-only claim can truncate a Bengali one). This module handles
that with three layers, not one:
  1. A generous token budget (config.LLM_MAX_TOKENS_VERIFICATION) and
     low temperature (config.LLM_TEMPERATURE_VERIFICATION) make
     truncation unlikely in the first place -- low temperature keeps
     responses concise and on-instruction rather than rambling.
  2. Truncation is DETECTED explicitly (finish_reason indicates the
     token limit was hit) before attempting to parse, instead of
     surfacing as a confusing generic JSONDecodeError.
  3. Any failure (truncation, malformed JSON, missing keys, invalid
     Prediction label) triggers a fresh retry -- up to
     config.MAX_VERIFICATION_RETRIES times -- rather than trying to
     "repair" a truncated response. Deliberately NOT attempting
     content-level repair (e.g. guessing how a cut-off sentence would
     have ended): patching in invented text is itself a form of the
     hallucination this whole stage is designed to avoid. A clean
     retry from scratch is the only safe fix for a truncated response.

Uses GPT-4o mini via a SELF-CONTAINED call here (not reusing
query_generation.py) -- this stage must not touch or depend on any
part of the existing retrieval/fusion pipeline, per the requirement
that only files necessary for this new stage are modified.
"""
import json
import re

from . import article_fetcher, config

PROMPT_TEMPLATE = """You are a careful fact-checking assistant.

STRICT RULES:
- Reason ONLY using the evidence text provided below (the top-3 retrieved sources).
- Do NOT use any outside/background knowledge you may have.
- Do NOT invent, assume, or infer facts that are not explicitly stated in the evidence.
- Be concise and stay within the word limits below -- do not pad or ramble.

DECISION RULE (binary, no third option exists):
- Your decision must be based ONLY on what the retrieved evidence actually says --
  never on an assumption, a default rule, or the mere ABSENCE of evidence.
- Output "SUPPORTS" if the retrieved evidence directly corroborates the claim's core
  assertion.
- Output "REFUTES" if, having genuinely examined the retrieved evidence, it either
  (a) directly contradicts the claim's core assertion, or (b) fails to substantiate
  it on the merits of what the evidence actually states.
- Do NOT output "REFUTES" merely because evidence is sparse, missing, or you were
  unable to find corroboration -- the absence of supporting evidence is not itself
  a reason to refute. Your label must reflect a genuine reading of what the
  retrieved evidence says, not a mechanical default triggered by thin evidence.
- There is no third label. Do not hedge: commit to one of the two labels based on
  your actual assessment of the evidence, not on how much of it there is.
- NEVER use hedging phrases anywhere in your response, including but not limited to:
  "insufficient evidence", "evidence is incomplete", "cannot verify", "not enough
  evidence", "unable to verify", "cannot determine", "unclear from the evidence"
  (or their equivalent in whatever language you are writing in). These phrases are
  forbidden in both the Evidence and Justification fields. Instead, state directly
  what the evidence does and does not establish, and commit to a label.

GROUNDING AND NO-HALLUCINATION REQUIREMENT:
- Every claim you make in "Evidence" and "Justification" must be traceable to the
  retrieved evidence text provided below. Do not introduce any fact, name, date,
  number, or detail that is not explicitly present in that evidence.
- The "Justification" field must explicitly state that its conclusion is drawn from
  the retrieved evidence (e.g. reference "the retrieved evidence" or "Source 1/2/3"
  directly), so it is clear the decision is grounded in what was retrieved and not
  in outside knowledge.

OUTPUT LANGUAGE (applies ONLY to the "Evidence" and "Justification" fields):
{output_language_instruction}

CRITICAL -- "Prediction" FIELD IS THE ONE EXCEPTION TO THE RULE ABOVE:
Regardless of what language you write "Evidence" and "Justification" in, the
"Prediction" field itself must ALWAYS be written in English, and must be EXACTLY
one of the two literal strings "SUPPORTS" or "REFUTES" -- never translated,
transliterated, or written in any other script, no matter what language the claim
or your other two fields are in.

Claim (language: {lang}):
{claim_text}

Evidence retrieved for this claim (top-3 fusion-ranked, full-page content):
{evidence_block}

Based ONLY on the evidence above, respond in STRICT JSON only, no other text, no
markdown fences, with EXACTLY these three keys:

1. "Evidence": a structured, concise summary of what the retrieved evidence actually
   says, grounded only in the text above. Where useful, indicate which source
   supports which specific point. Write this field in the OUTPUT LANGUAGE specified
   above. STRICT LENGTH: 100-150 words, no more.
2. "Prediction": exactly one of the two strings "SUPPORTS" or "REFUTES", written in
   English exactly as shown -- this exact spelling and capitalization, nothing else,
   no other label, no extra words, and NEVER translated into the claim's language.
   Follow the DECISION RULE above.
3. "Justification": explains why the evidence supports or refutes the claim, based
   only on the retrieved evidence, citing specific details from it. Must directly
   state the reasoning, not hedge. Write this field in the OUTPUT LANGUAGE specified
   above. STRICT LENGTH: exactly 100-120 words, no more, no less.

Respond as JSON: {{"Evidence": "...", "Prediction": "...", "Justification": "..."}}
"""

# ---------------------------------------------------------------------
# Output-language instruction, derived dynamically from the claim's
# `lang` tag (set upstream by data_loader.py / query_generation.py --
# one of "english", "hindi", "bengali", "codemix"). Nothing here is
# hardcoded to a specific claim; this runs identically for claim #1 or
# claim #1009, driven only by that claim's own lang field.
#
# BUG FIX (2026-07): the prompt previously only mentioned the claim's
# language as passive context ("Claim (language: {lang})") without ever
# instructing the model what language to WRITE its answer in. Verified
# on real output: a Bengali claim (S2/T/BN/1001) produced fully English
# Evidence/Justification text. Gemini was never told the output should
# match the input language -- it defaulted to English regardless of
# `lang`. Fixed by adding an explicit, unambiguous instruction, and
# additionally reinforcing that "Prediction" is the ONE field that must
# NOT follow this instruction (it must stay English SUPPORTS/REFUTES
# regardless of what language the other two fields are written in).
# ---------------------------------------------------------------------
_OUTPUT_LANGUAGE_INSTRUCTIONS = {
    "english": (
        'Write the "Evidence" and "Justification" fields in English.'
    ),
    "hindi": (
        'Write the "Evidence" and "Justification" fields ENTIRELY in Hindi '
        "(Devanagari script). Do not switch to English or transliterate into "
        "Roman script anywhere in these two fields."
    ),
    "bengali": (
        'Write the "Evidence" and "Justification" fields ENTIRELY in Bengali '
        "(Bengali script). Do not switch to English or transliterate into "
        "Roman script anywhere in these two fields."
    ),
    "codemix": (
        'Write the "Evidence" and "Justification" fields in the SAME code-mixed '
        "style as the original claim -- naturally mixing English with Hindi/Bengali "
        "the way the claim itself does, rather than translating fully into a single "
        "language. Match the claim's own register and script-mixing pattern."
    ),
}

_DEFAULT_OUTPUT_LANGUAGE_INSTRUCTION = (
    'Write the "Evidence" and "Justification" fields in the SAME language as the '
    "claim text above."
)


def _get_output_language_instruction(lang: str) -> str:
    """Looks up the correct instruction for whatever `lang` this claim
    happens to have (any of the 1009 claims, any of the 4 languages) --
    falls back to a generic same-language instruction if `lang` is ever
    something unexpected, rather than silently defaulting to English."""
    return _OUTPUT_LANGUAGE_INSTRUCTIONS.get(
        lang.lower().strip(), _DEFAULT_OUTPUT_LANGUAGE_INSTRUCTION
    )

# Checked against Evidence + Justification after generation -- even with
# the prompt's explicit ban above, a low-but-nonzero temperature model can
# still occasionally slip a hedge phrase in. Caught here and treated as a
# failed attempt (same retry path as truncation/malformed JSON), since
# FIRE 2026 has no "not enough evidence" label -- a hedge phrase in the
# text means the model didn't actually commit to the binary decision it
# was required to make.
BANNED_HEDGE_PHRASES = (
    "insufficient evidence",
    "evidence is incomplete",
    "incomplete evidence",
    "cannot verify",
    "can not verify",
    "unable to verify",
    "not enough evidence",
    "insufficient information",
    "cannot determine",
    "unclear from the evidence",
    "no clear evidence",
    "not clear whether",
)

STRICT_RETRY_SUFFIX = """

IMPORTANT: Your previous attempt did not produce a valid, complete response.
Follow the JSON format and word limits EXACTLY: valid JSON only, no markdown
fences, "Prediction" must be exactly "SUPPORTS" or "REFUTES" (nothing else),
"Evidence" must be 100-150 words, "Justification" must be exactly 100-120
words. Be concise -- do not run long. Respond again from scratch.
"""

REQUIRED_KEYS = ("Evidence", "Prediction", "Justification")

# CHANGED (Gemini -> GPT-4o mini): renamed from `_gemini_model`.
_gemini_client = None


def _get_gemini_client():
    """Self-contained Gemini client for this stage only -- deliberately
    not shared with query_generation.py, so this stage has zero
    coupling to the existing retrieval pipeline's internals.

    CHANGED AGAIN (2026-07): the `google.generativeai` package used in
    the first Gemini swap-back is now a FULLY DEPRECATED SDK (see
    https://github.com/google-gemini/deprecated-generative-ai-python) --
    it's what caused the "gemini-2.5-flash is no longer available to
    new users" 404, since new API keys/projects aren't provisioned
    against the model listing that old SDK/endpoint expects. This now
    uses Google's current, actively-maintained `google-genai` package
    (`from google import genai`) instead -- a different import AND a
    different client-construction pattern (genai.Client(...) instead of
    genai.configure() + GenerativeModel(...)), but the same role in this
    file: a lazily-constructed, cached client returned to callers below.
    """
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add it to your .env file or "
                "export it as an environment variable.\n"
                "Get a key at https://aistudio.google.com/apikey"
            )
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def _fetch_evidence_texts(top_evidence: list) -> list:
    """For each evidence item, fetch the full article text. Falls back
    to the item's existing `snippet` if fetching/extraction returns
    nothing (dead link, paywall, blocked scrape, etc.) -- so a fetch
    failure degrades quality rather than dropping the source entirely.

    Returns a list of {"title", "url", "text", "used_fallback_snippet"}.
    """
    results = []
    for item in top_evidence:
        url = item.get("url", "")
        title = item.get("title", "")
        snippet = item.get("snippet", "")

        full_text = article_fetcher.fetch_and_extract(url) if url else ""
        used_fallback = not full_text.strip()
        text = full_text if full_text.strip() else snippet

        results.append({
            "title": title, "url": url, "text": text,
            "used_fallback_snippet": used_fallback,
        })
    return results


def _build_evidence_block(evidence_texts: list) -> str:
    blocks = []
    for i, ev in enumerate(evidence_texts, start=1):
        blocks.append(
            f"[Source {i}] {ev['title']}\nURL: {ev['url']}\n{ev['text']}\n"
        )
    return "\n".join(blocks)


def _clean_json_text(raw_text: str) -> str:
    """Light, content-safe cleanup only -- never invents or repairs
    missing content. Handles two harmless cosmetic issues that can slip
    through even with response_mime_type='application/json':
      - stray markdown fences
      - a trailing comma before a closing brace/bracket
    Does NOT attempt to fix a genuinely truncated/incomplete string --
    that's detected separately (see _is_truncated) and handled by
    retrying the whole generation, not by patching the text.
    """
    cleaned = raw_text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
    return cleaned


def _is_truncated(resp) -> bool:
    """True if the model's response was cut off by hitting the token
    limit mid-generation, rather than finishing normally. Checked before
    parsing so truncation produces a clear, specific error instead of a
    confusing generic JSONDecodeError from a chopped-off string.

    CHANGED (deprecated google.generativeai -> current google-genai
    SDK): `resp.candidates[0].finish_reason` is now a `FinishReason`
    enum whose `.name` is the string "MAX_TOKENS" (rather than the old
    SDK's bare integer value 2) -- comparing by `.name` is more robust
    across SDK versions than relying on a specific enum identity/value.
    """
    if not resp.candidates:
        return False
    reason = resp.candidates[0].finish_reason
    return getattr(reason, "name", str(reason)) == "MAX_TOKENS"


def _find_hedge_phrase(text: str) -> str:
    """Returns the first banned hedge phrase found in text (case-insensitive
    substring match), or "" if none. Checked against both Evidence and
    Justification -- FIRE 2026 has no 'not enough evidence' label, so any
    hedge phrase in the text means the model didn't actually commit to
    the required binary decision."""
    lower = text.lower()
    for phrase in BANNED_HEDGE_PHRASES:
        if phrase in lower:
            return phrase
    return ""


# Multilingual keyword variants for the grounding-reference check below --
# output can legitimately be in any of the 4 claim languages (per
# _get_output_language_instruction), so an English-only keyword check
# would always report a false "missing" note for Hindi/Bengali output.
_GROUNDING_REFERENCE_KEYWORDS = (
    "evidence", "source",                       # english
    "साक्ष्य", "स्रोत", "सबूत",                    # hindi
    "প্রমাণ", "সাক্ষ্য", "সূত্র",                   # bengali
)


def _references_retrieved_evidence(text: str) -> bool:
    """Soft, non-blocking check (informational NOTE only, does not trigger
    a retry) that the Justification explicitly grounds its conclusion in
    the retrieved evidence, per the grounding requirement in the prompt.
    Case-insensitive substring match across English/Hindi/Bengali keyword
    variants -- CodeMix claims typically mix English in, so the English
    keywords already cover that case too."""
    lower = text.lower()
    return any(kw.lower() in lower for kw in _GROUNDING_REFERENCE_KEYWORDS)


def _call_llm_once(prompt: str) -> dict:
    """A single LLM call + parse attempt. Raises ValueError with a
    specific, diagnosable message on any failure (truncation, malformed
    JSON, missing keys, invalid Prediction, or hedge language) -- never
    returns a partial/guessed/hedging result.

    CHANGED (deprecated google.generativeai -> current google-genai
    SDK): the prompt string, the parsing logic, and every validation
    rule below are UNCHANGED; only the API call itself changed.

    CHANGED AGAIN (2026-07): the earlier version of this function also
    sent `thinking_config=types.ThinkingConfig(thinking_budget=0)`, with
    a fallback that retried without it if the error message contained
    the word "thinking". In practice, this model returns a generic
    "Request contains an invalid argument" with NO further detail on
    400s, so that string-matching fallback never actually triggered --
    the same 400 just repeated on every retry. Rather than keep guessing
    at which specific parameter this model generation rejects,
    thinking_config is removed entirely here. This trades away a
    speculative fix for a THEORETICAL problem (thinking silently eating
    the token budget) in order to resolve a CONFIRMED one (every call
    failing outright). LLM_MAX_TOKENS_VERIFICATION was already raised to
    8192 (see config.py) as a headroom cushion in case thinking stays on
    by default and consumes part of the budget.

    Also now prints much more of the raw exception on failure (message,
    args, and any `response`/`details` attributes the SDK's exception
    object exposes) instead of just str(e) -- so if a 400 happens again,
    the terminal output actually has enough detail to diagnose the real
    field/value at fault, rather than the same uninformative message
    repeating three times.
    """
    from google.genai import types

    client = _get_gemini_client()
    try:
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=config.LLM_TEMPERATURE_VERIFICATION,
                max_output_tokens=config.LLM_MAX_TOKENS_VERIFICATION,
                response_mime_type="application/json",
            ),
        )
    except Exception as e:
        extra = {
            attr: getattr(e, attr)
            for attr in ("response", "details", "status", "code", "args")
            if hasattr(e, attr)
        }
        raise ValueError(
            f"Gemini API call failed: {type(e).__name__}: {e}\n"
            f"  Additional exception detail (for diagnosis): {extra}"
        ) from e

    if _is_truncated(resp):
        raise ValueError(
            "Response was cut off (hit max_tokens = "
            f"{config.LLM_MAX_TOKENS_VERIFICATION}). Bengali/Hindi text tokenizes "
            "less efficiently than English, which makes this more likely for "
            "non-English claims."
        )

    raw_text = (resp.text or "").strip()
    if not raw_text:
        raise ValueError(
            "Empty response text from Gemini (finish_reason="
            f"{getattr(resp.candidates[0].finish_reason, 'name', None) if resp.candidates else None})."
        )
    cleaned = _clean_json_text(raw_text)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON ({e}). Raw response:\n{raw_text}")

    missing = [k for k in REQUIRED_KEYS if k not in result]
    if missing:
        raise ValueError(f"Response missing expected key(s) {missing}: {result}")

    if result.get("Prediction") not in config.VALID_PREDICTIONS:
        raise ValueError(
            f"Invalid Prediction label {result.get('Prediction')!r} -- "
            f"must be exactly one of {config.VALID_PREDICTIONS}."
        )

    for field in ("Evidence", "Justification"):
        hedge = _find_hedge_phrase(result.get(field, ""))
        if hedge:
            raise ValueError(
                f"Hedge phrase {hedge!r} found in {field!r} -- FIRE 2026 requires a "
                f"decisive SUPPORTS/REFUTES call with no hedging language."
            )

    return result


def _call_llm_with_retries(prompt: str, claim_id: str, max_retries: int = None) -> dict:
    """Unified retry wrapper -- covers truncation, malformed JSON,
    missing keys, and an invalid Prediction label all in one place,
    since all four are "this attempt didn't produce a valid, usable
    result" and the correct response to every one of them is the same:
    a fresh generation attempt, never a patched/guessed result.

    Total attempts = 1 + max_retries (default: config.MAX_VERIFICATION_RETRIES,
    i.e. up to 3 attempts total). Raises RuntimeError with the last
    underlying error if every attempt fails.

    CHANGED (Gemini -> GPT-4o mini): renamed from
    `_call_gemini_with_retries` (private helper, not called from
    outside this file). Retry count/logic/backoff behavior UNCHANGED --
    only the inner call (`_call_llm_once`) and the error message text
    were updated.
    """
    max_retries = config.MAX_VERIFICATION_RETRIES if max_retries is None else max_retries

    last_error = None
    for attempt in range(1, max_retries + 2):
        try:
            attempt_prompt = prompt if attempt == 1 else prompt + STRICT_RETRY_SUFFIX
            return _call_llm_once(attempt_prompt)
        except Exception as e:
            last_error = e
            print(f"[verification] WARNING: attempt {attempt}/{max_retries + 1} "
                  f"failed for claim {claim_id}: {e}")

    raise RuntimeError(
        f"Claim {claim_id}: GPT-4o mini failed to produce a valid response after "
        f"{max_retries + 1} attempts. Last error: {last_error}"
    )


def verify_claim(claim: dict, top_evidence: list) -> dict:
    """claim: {"id": str, "text": str, "lang": str}
    top_evidence: the top-N items from fusion_ranked_evidence (each
    with at least title/url/snippet).

    Returns the exact FIRE 2026 Subtask-2 record:
        {"ID": ..., "Evidence": ..., "Prediction": ..., "Justification": ...}
    """
    claim_id, claim_text, lang = claim["id"], claim["text"], claim["lang"]

    evidence_texts = _fetch_evidence_texts(top_evidence)
    evidence_block = _build_evidence_block(evidence_texts)

    output_language_instruction = _get_output_language_instruction(lang)
    prompt = PROMPT_TEMPLATE.format(
        lang=lang,
        claim_text=claim_text,
        evidence_block=evidence_block,
        output_language_instruction=output_language_instruction,
    )

    result = _call_llm_with_retries(prompt, claim_id)  # CHANGED: was _call_gemini_with_retries

    evidence_word_count = len(result["Evidence"].split())
    if not (100 <= evidence_word_count <= 150):
        print(f"[verification] NOTE: claim {claim_id} evidence summary is "
              f"{evidence_word_count} words (target: 100-150) -- accepted, not blocking.")

    justification_word_count = len(result["Justification"].split())
    if not (100 <= justification_word_count <= 120):
        print(f"[verification] NOTE: claim {claim_id} justification is "
              f"{justification_word_count} words (target: 100-120) -- accepted, not blocking.")

    if not _references_retrieved_evidence(result["Justification"]):
        print(f"[verification] NOTE: claim {claim_id} justification does not "
              f"explicitly reference the retrieved evidence (e.g. 'the evidence', "
              f"'Source 1') -- accepted, not blocking, but worth a manual look.")

    return {
        "ID": claim_id,
        "Evidence": result["Evidence"],
        "Prediction": result["Prediction"],
        "Justification": result["Justification"],
    }


def verify_claim_folder(
    claim_folder,
    fusion_filename: str = None,
    output_filename: str = None,
    top_n: int = None,
) -> dict:
    """Convenience wrapper for running this stage on ONE claim's output
    folder -- same convention as every prior stage. claim_folder is
    supplied by the caller; nothing here is hardcoded to a specific claim.

    Reads:  <claim_folder>/fusion_ranked.json   (for claim + fusion_ranked_evidence)
    Writes: <claim_folder>/verification_output.json  (exact FIRE 2026 record)
    """
    from pathlib import Path

    folder = Path(claim_folder)
    fusion_filename = fusion_filename or config.FUSION_OUTPUT_FILENAME
    output_filename = output_filename or config.VERIFICATION_OUTPUT_FILENAME
    top_n = top_n or config.TOP_N_FOR_VERIFICATION

    with open(folder / fusion_filename, encoding="utf-8") as f:
        fusion_data = json.load(f)

    claim_meta = fusion_data["claim"]
    fusion_ranked_evidence = fusion_data["fusion_ranked_evidence"]
    top_evidence = sorted(fusion_ranked_evidence, key=lambda x: x["fusion_rank"])[:top_n]

    record = verify_claim(claim_meta, top_evidence)

    output_path = folder / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"[verification] Saved verification record -> {output_path}")
    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the final LLM verification stage on one claim's output folder. "
                    "No claim ID or file path is hardcoded -- pass the folder to process. "
                    "Requires OPENAI_API_KEY set."  # CHANGED: was "Requires GEMINI_API_KEY set."
    )
    parser.add_argument(
        "claim_folder",
        help="Path to a single claim's output folder, e.g. outputs/claims/S2_T_BN_1001 "
             "(must already contain fusion_ranked.json).",
    )
    args = parser.parse_args()

    verify_claim_folder(args.claim_folder)