"""
Groq-based draft refiner.

Input: raw script built from source_1 summary.
Output: cleaned script JSON with same shape.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_PROMPT_VERSION = "v5"
LLM_INPUT_SECTION_LIMITS = {
    "short": 4,
    "medium": 6,
    "long": 8,
}
LLM_INPUT_BLOCK_CHAR_LIMITS = {
    "short": 380,
    "medium": 520,
    "long": 720,
}
LLM_INPUT_TOTAL_CHAR_LIMITS = {
    "short": 1400,
    "medium": 2400,
    "long": 3600,
}


def refine_draft_with_groq(
    raw_script: dict[str, Any],
    language: str,
    style: str | None = None,
    prompt_override: str | None = None,
    target_blocks: int = 5,
    length_target: str = "medium",
    timeout_sec: float = 25.0,
) -> dict[str, Any]:
    """
    Refine a raw script via Groq Chat Completions API.

    Raises RuntimeError on any API or parsing failure.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    model = (os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL)
    normalized_length = _normalize_length_target(length_target)
    llm_input_script, llm_input_meta = _build_llm_input_script(
        raw_script=raw_script,
        target_blocks=target_blocks,
        length_target=normalized_length,
    )
    prompt, prompt_mode, prompt_version = _build_prompt(
        raw_script=llm_input_script,
        language=language,
        style=style,
        prompt_override=prompt_override,
        target_blocks=target_blocks,
        length_target=normalized_length,
    )

    from ..llm_client import chat_completion

    try:
        content = chat_completion(
            system=(
                "You rewrite short narration drafts and return strict JSON only. "
                "Do not add facts that are not in the source text."
            ),
            user=prompt,
            model=model,
            temperature=0.3,
            timeout=timeout_sec,
        )
    except Exception as exc:
        raise RuntimeError(_normalize_groq_error(exc)) from exc

    if not content:
        raise RuntimeError("LLM returned empty content.")

    refined_script = _parse_script_json(content)
    _validate_script_shape(refined_script)

    return {
        "script": refined_script,
        "meta": {
            "provider": "groq+gemini-fallback",
            "model": model,
            "prompt_version": prompt_version,
            "prompt_mode": prompt_mode,
            "target_blocks": max(1, int(target_blocks)),
            "length_target": normalized_length,
            "llm_input_blocks": llm_input_meta["selected_blocks"],
            "llm_input_chars": llm_input_meta["selected_chars"],
            "source_blocks": llm_input_meta["source_blocks"],
            "source_chars": llm_input_meta["source_chars"],
        },
    }


def _build_prompt(
    raw_script: dict[str, Any],
    language: str,
    style: str | None,
    prompt_override: str | None = None,
    target_blocks: int = 5,
    length_target: str = "medium",
) -> tuple[str, str, str]:
    normalized_length = _normalize_length_target(length_target)
    normalized_blocks = max(1, int(target_blocks))
    sentence_rule, length_hint = _length_rules(normalized_length)
    override = str(prompt_override or "").strip()
    if override:
        prompt = (
            f"{override}\n\n"
            "Output requirements:\n"
            "1) Keep same language as requested.\n"
            "2) Do not invent facts.\n"
            "3) Return strict JSON only: {\"language\":\"...\", \"blocks\":[{\"text\":\"...\"}, ...]}.\n"
            f"4) Target length profile: {normalized_length} ({length_hint}).\n"
            f"5) Target block count: {normalized_blocks}. Keep this count when possible using source facts only.\n"
            f"6) {sentence_rule}\n"
            "7) Start with a strong curiosity-driven hook, not a generic introduction.\n"
            "8) Front-load the most surprising, counterintuitive, or high-stakes fact early.\n"
            "9) Use short, punchy, conversational lines that sound natural in voice-over.\n"
            "10) Make each block add new information or escalate the stakes.\n"
            "11) Create curiosity gaps so the viewer wants the next line.\n"
            "12) Alternate short punch lines with slightly longer explanation lines for rhythm.\n"
            "13) Avoid filler, repetition, generic intros, bland documentary phrasing, and weak transitions.\n"
            "14) Do not add fake details, fake numbers, fake motives, fake quotes, or certainty not supported by the source.\n"
            f"Requested language: {language}\n"
            f"Raw draft JSON:\n{json.dumps(raw_script, ensure_ascii=False)}"
        )
        return prompt, "override", "custom"

    style_hint = (
        f"Style hint: {style}."
        if style
        else "Style hint: fast, curiosity-driven short-form narration with sharp pacing and spoken rhythm."
    )
    prompt = (
        "Rewrite the following raw draft into high-retention short-video narration blocks.\n"
        "Rules:\n"
        "1) Keep the same language as requested.\n"
        "2) Do not invent facts.\n"
        "3) Return strict JSON only with this shape: "
        '{"language":"...", "blocks":[{"text":"..."}, ...]}.\n'
        f"4) Target length profile: {normalized_length} ({length_hint}).\n"
        f"5) Target block count: {normalized_blocks}. Keep this count when possible using source facts only.\n"
        f"6) {sentence_rule}\n"
        "7) Start with a strong curiosity-driven hook. Do not start with generic framing like 'In a catastrophic event' or 'In history'.\n"
        "8) Front-load the most surprising, counterintuitive, or hard-to-believe fact early.\n"
        "9) Turn the material into a forward-moving sequence: hook -> tension -> reveal -> payoff.\n"
        "10) Every block must add new information or raise the stakes. Do not repeat the same impact in different words.\n"
        "11) Use short, punchy, conversational sentences that sound good spoken aloud.\n"
        "12) Create curiosity gaps between blocks so the viewer wants the next line.\n"
        "13) Alternate rhythm: mix very short punch lines with slightly longer explanation lines.\n"
        "14) Prefer concrete details, striking contrasts, unusual causes, and surprising consequences when supported by the source.\n"
        "15) If a fact is uncertain or nuanced in the source, keep that nuance instead of overstating it.\n"
        "16) Avoid bland documentary-summary tone, filler, repetition, generic intros, and weak transitions.\n"
        "17) End with a strong final beat, not a bland summary.\n"
        "18) Bad opening style: 'In a catastrophic event that would change history...'\n"
        "19) Better opening style: lead with the shocking cause, shocking number, or unbelievable consequence.\n"
        f"Requested language: {language}\n"
        f"{style_hint}\n"
        f"Raw draft JSON:\n{json.dumps(raw_script, ensure_ascii=False)}"
    )
    return prompt, "default", DEFAULT_PROMPT_VERSION


def _normalize_length_target(length_target: str) -> str:
    raw = str(length_target or "").strip().lower()
    if raw in {"short", "medium", "long"}:
        return raw
    return "medium"


def _length_rules(length_target: str) -> tuple[str, str]:
    if length_target == "short":
        return (
            "Keep each block to 1 short sentence.",
            "very concise summary, minimal detail",
        )
    if length_target == "long":
        return (
            "Keep each block 2-3 sentences with richer factual detail and smooth transitions.",
            "deeper explanation while staying factual",
        )
    return (
        "Keep each block 1-2 short sentences with balanced detail.",
        "balanced pacing and detail",
    )


def _normalize_groq_error(exc: Exception) -> str:
    text = str(exc).strip()
    if text:
        return text
    return f"Groq SDK error: {type(exc).__name__}"


def _build_llm_input_script(
    raw_script: dict[str, Any],
    *,
    target_blocks: int,
    length_target: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    source_blocks = list(raw_script.get("blocks") or [])
    normalized_length = _normalize_length_target(length_target)
    source_chars = sum(len(str(block.get("text") or "").strip()) for block in source_blocks if isinstance(block, dict))

    max_sections = max(
        1,
        min(
            len(source_blocks),
            max(int(target_blocks), LLM_INPUT_SECTION_LIMITS[normalized_length]),
        ),
    )
    per_block_char_limit = LLM_INPUT_BLOCK_CHAR_LIMITS[normalized_length]
    total_char_limit = LLM_INPUT_TOTAL_CHAR_LIMITS[normalized_length]

    selected_blocks: list[dict[str, Any]] = []
    selected_chars = 0
    for block in source_blocks:
        if not isinstance(block, dict):
            continue
        text = _trim_text_to_chars(str(block.get("text") or "").strip(), per_block_char_limit)
        if not text:
            continue

        remaining_chars = total_char_limit - selected_chars
        if remaining_chars <= 0:
            break
        if len(text) > remaining_chars:
            text = _trim_text_to_chars(text, remaining_chars)
        if not text:
            break

        next_block = dict(block)
        next_block["text"] = text
        selected_blocks.append(next_block)
        selected_chars += len(text)
        if len(selected_blocks) >= max_sections:
            break

    if not selected_blocks:
        fallback_text = _trim_text_to_chars(
            " ".join(
                str(block.get("text") or "").strip()
                for block in source_blocks
                if isinstance(block, dict)
            ).strip(),
            total_char_limit,
        )
        if fallback_text:
            selected_blocks = [{"text": fallback_text}]
            selected_chars = len(fallback_text)

    llm_input_script = {
        "language": raw_script.get("language") or "vi-VN",
        "blocks": selected_blocks,
    }
    llm_input_meta = {
        "source_blocks": len(source_blocks),
        "source_chars": source_chars,
        "selected_blocks": len(selected_blocks),
        "selected_chars": selected_chars,
    }
    return llm_input_script, llm_input_meta


def _trim_text_to_chars(text: str, limit: int) -> str:
    clean = str(text or "").strip()
    if not clean or limit <= 0:
        return ""
    if len(clean) <= limit:
        return clean
    clipped = clean[:limit].rsplit(" ", 1)[0].strip()
    if len(clipped) < max(40, limit // 2):
        clipped = clean[:limit].strip()
    return clipped


def _parse_script_json(content: str) -> dict[str, Any]:
    """
    Parse strict JSON; if wrapped in markdown fence, extract inner JSON.
    """
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # Fallback: slice first JSON object boundaries if model added preface.
    if not text.startswith("{"):
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            text = text[first : last + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM output is not valid JSON script.") from exc


def _validate_script_shape(script: dict[str, Any]):
    if not isinstance(script, dict):
        raise RuntimeError("Refined script must be a JSON object.")
    if not isinstance(script.get("language"), str) or not script.get("language").strip():
        raise RuntimeError("Refined script is missing valid 'language'.")
    blocks = script.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise RuntimeError("Refined script must include non-empty 'blocks'.")
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise RuntimeError(f"Block {idx} is not an object.")
        text = block.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"Block {idx} has invalid text.")
