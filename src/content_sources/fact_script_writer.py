"""
Fact-to-script writer for short-form video narration.

Converts ranked fact candidates into a structured narration script
with role-assigned blocks (hook, setup, twist, reveal, consequence, payoff).
Uses Groq LLM for generation with anti-repetition validation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path as _Path
from typing import Any

from pydantic import BaseModel, Field

from src.prompts import render as _render_prompt

_PROMPT_DIR = _Path(__file__).parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")

logger = logging.getLogger(__name__)

BLOCK_ROLES = ("hook", "setup", "twist", "reveal", "consequence", "payoff")

STRONG_ADJECTIVES = frozenset({
    "staggering",
    "devastating",
    "massive",
    "haunting",
    "catastrophic",
    "unprecedented",
    "deadly",
    "horrifying",
})

_NUMBER_RE = re.compile(r"\b\d[\d,.]+\b")
_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)

COST_SCALE_PATTERNS = re.compile(
    r"\$[\d,.]+\s*(billion|million|trillion|thousand)|"
    r"\d[\d,.]*\s*(billion|million|trillion|thousand)\s*(dollars|people|tons|gallons)",
    re.IGNORECASE,
)


class RankedFact(BaseModel):
    fact_text: str
    hook_text: str = ""
    source_url: str = ""
    score: float = 0.0
    suggested_role: str = ""
    reason_tags: list[str] = Field(default_factory=list)
    fact_id: str = ""


def write_script_from_facts(
    facts: list[dict],
    *,
    language: str = "en-US",
    target_blocks: int = 6,
    style: str | None = None,
    timeout_sec: float = 45.0,
    extra_constraints: list[str] | None = None,
    skill: dict | None = None,
    video_goal: str = "",
    must_cover: str = "",
) -> dict:
    """
    Convert ranked facts into a narration script.

    Returns {"script": {"language": ..., "blocks": [...]}, "meta": {...}}.
    Falls back to direct assembly when LLM call fails entirely.
    """
    parsed_facts = _parse_facts(facts)
    if not parsed_facts:
        return _empty_result(language)

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not groq_key and not gemini_key:
        raise RuntimeError("Script LLM unavailable: no GROQ_API_KEY or GEMINI_API_KEY set.")

    try:
        script, meta = _generate_via_llm(
            parsed_facts,
            language=language,
            target_blocks=target_blocks,
            style=style,
            timeout_sec=timeout_sec,
            extra_constraints=extra_constraints,
            skill=skill,
            video_goal=video_goal,
            must_cover=must_cover,
        )
    except Exception as exc:
        raise RuntimeError(f"Script LLM unavailable: {exc}") from exc

    # Skip anti-repetition retry if caller already provided constraints
    if extra_constraints:
        return {"script": script, "meta": meta}

    warnings: list[str] = []
    issues = _validate_anti_repetition(script)
    if issues:
        logger.info("Anti-repetition issues found, regenerating: %s", issues)
        try:
            script, retry_meta = _generate_via_llm(
                parsed_facts,
                language=language,
                target_blocks=target_blocks,
                style=style,
                timeout_sec=timeout_sec,
                extra_constraints=issues,
                skill=skill,
                video_goal=video_goal,
                must_cover=must_cover,
            )
            meta = retry_meta
        except Exception as exc:
            logger.warning("Retry generation failed: %s", exc)
            warnings.append("Anti-repetition retry failed; returning best attempt.")

        retry_issues = _validate_anti_repetition(script)
        if retry_issues:
            warnings.append(
                f"Script still has repetition issues after retry: {retry_issues}"
            )

    meta["warnings"] = warnings
    return {"script": script, "meta": meta}


def _parse_facts(raw: list[dict]) -> list[RankedFact]:
    results: list[RankedFact] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            results.append(RankedFact(**item))
        except Exception:
            if item.get("fact_text"):
                results.append(RankedFact(fact_text=str(item["fact_text"])))
    return results


def _empty_result(language: str) -> dict:
    return {
        "script": {"language": language, "blocks": []},
        "meta": {"provider": "none", "warnings": ["No facts provided."]},
    }


def _build_system_prompt(
    facts: list[RankedFact],
    *,
    language: str,
    target_blocks: int,
    style: str | None,
    extra_constraints: list[str] | None = None,
    skill: dict | None = None,
    video_goal: str = "",
    must_cover: str = "",
) -> str:
    structure = skill.get("structure", {}) if skill else {}
    is_vietnamese = language.startswith("vi")

    if skill and skill.get("structure", {}).get("tone"):
        skill_tone = skill["structure"]["tone"]
    elif style:
        skill_tone = style
    else:
        skill_tone = "fast, curiosity-driven short-form narration with sharp pacing and spoken rhythm."

    hook_rule = structure.get("hook_rule") or (
        "Scan ALL the facts. Find the MOST SHOCKING one (a death count, a dark secret, "
        "a betrayal, a mind-blowing number). Put it FIRST.\n"
        "- The first sentence must make viewers think 'WHAT?! I need to know more.'\n"
        "- TEMPLATE: '[Shocking statement]. [Question that creates curiosity]. [Promise of revelation].'"
    )

    pacing_rule = structure.get("pacing_rule") or (
        "Alternate sentence length: short punch, then longer context, then short punch.\n"
        "- Every 6 seconds of audio (roughly 15-20 words), there should be a new revelation or twist."
    )

    if skill:
        example_hook = skill.get("example_hook_vi" if is_vietnamese else "example_hook_en", "")
    else:
        example_hook = ""
    if not example_hook:
        example_hook = (
            "Good: 'This city was abandoned overnight. 350,000 people just... left.'"
        )
    # Substitute [topic] placeholder with the actual topic so the LLM doesn't copy it literally
    _topic_sub = video_goal or "the topic"
    example_hook = example_hook.replace("[topic]", _topic_sub).replace("[myth]", _topic_sub).replace("[theory]", _topic_sub)
    hook_rule = hook_rule.replace("[topic]", _topic_sub).replace("[year]", "the year")

    # Build citation requirement block
    facts_with_ids = [f for f in facts if f.fact_id]
    if facts_with_ids:
        fact_list_with_ids = "\n".join(
            f"{f.fact_id}: {f.fact_text[:150]}" for f in facts_with_ids
        )
        citation_requirement = (
            "CITATION REQUIREMENT:\n"
            "You are writing from research facts, each labeled with an ID like F001, F002, etc.\n"
            "At the END of every sentence that uses a fact, append the citation in square brackets:\n"
            '  "Subaru\'s ability lets him return to a save point. [F001]"\n'
            '  "The creator rewrote the opening arc. [F003, F004]"\n'
            "\nRules:\n"
            "- A sentence may cite multiple facts: [F001, F002].\n"
            "- Every declarative sentence in the body MUST cite at least one fact.\n"
            "- The hook (first sentence) and the closing loop-back sentence do not require citations.\n"
            "- Do NOT invent F-numbers that weren't provided.\n"
            "- Do NOT cite facts whose IDs are not in the provided list.\n"
            "\nFACTS (cite by ID):\n"
            f"{fact_list_with_ids}"
        )
    else:
        citation_requirement = ""

    skill_prompt_injection = (skill.get("prompt_injection", "") or "") if skill else ""

    return _render_prompt(
        "script_write",
        video_goal=video_goal or "",
        must_cover=must_cover or "",
        skill_tone=skill_tone,
        skill_hook_rule=hook_rule,
        skill_pacing_rule=pacing_rule,
        skill_example_hook=example_hook,
        skill_prompt_injection=skill_prompt_injection,
        citation_requirement=citation_requirement,
    )


def _generate_via_llm(
    facts: list[RankedFact],
    *,
    language: str,
    target_blocks: int,
    style: str | None,
    timeout_sec: float,
    extra_constraints: list[str] | None = None,
    skill: dict | None = None,
    video_goal: str = "",
    must_cover: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    system_prompt = _build_system_prompt(
        facts,
        language=language,
        target_blocks=target_blocks,
        style=style,
        extra_constraints=extra_constraints,
        skill=skill,
        video_goal=video_goal,
        must_cover=must_cover,
    )
    user_message = _build_user_message(
        facts,
        language=language,
        target_blocks=target_blocks,
        style=style,
        extra_constraints=extra_constraints,
        skill=skill,
    )

    from ..llm_client import chat_completion_with_meta

    resp = chat_completion_with_meta(
        system=system_prompt,
        user=user_message,
        stage="script",
        temperature=0.3,
        timeout=timeout_sec,
    )
    if not resp.text:
        raise RuntimeError("LLM returned empty content.")

    script = _parse_script_json(resp.text)
    _validate_script_shape(script)

    meta: dict[str, Any] = {
        "provider": resp.provider,
        "model": resp.model,
        "target_blocks": target_blocks,
        "input_facts": len(facts),
        "output_blocks": len(script.get("blocks", [])),
        "retried": extra_constraints is not None,
    }
    return script, meta


def _build_user_message(
    facts: list[RankedFact],
    *,
    language: str,
    target_blocks: int,
    style: str | None,
    extra_constraints: list[str] | None = None,
    skill: dict | None = None,
) -> str:
    facts_payload: list[dict[str, Any]] = []
    for i, f in enumerate(facts):
        entry: dict[str, Any] = {
            "id": i,
            "fact": f.fact_text,
        }
        if f.hook_text:
            entry["hook"] = f.hook_text
        if f.suggested_role:
            entry["suggested_role"] = f.suggested_role
        if f.reason_tags:
            entry["tags"] = f.reason_tags
        facts_payload.append(entry)

    constraint_block = ""
    if extra_constraints:
        constraint_block = (
            "\nADDITIONAL CONSTRAINTS (must fix these issues from previous attempt):\n"
            + "\n".join(f"- {c}" for c in extra_constraints)
            + "\n"
        )

    # Language instruction
    lang_block = (
        f"Requested language: {language}\n"
        f"CRITICAL: ALL block text MUST be written in {language}. "
        f"Even if the input facts are in English, you MUST translate and rewrite them in {language}. "
        f"The audience speaks {language}. Do NOT output English text when the requested language is not English.\n"
    )

    user_msg = (
        f"{lang_block}"
        f"{constraint_block}"
        f"\nINPUT FACTS:\n{json.dumps(facts_payload, ensure_ascii=False, indent=2)}"
    )

    # Append citation IDs when facts carry them
    facts_with_ids = [f for f in facts if f.fact_id]
    if facts_with_ids:
        fact_list_with_ids = "\n".join(
            f"{f.fact_id}: {f.fact_text[:150]}" for f in facts_with_ids
        )
        user_msg += (
            "\n\nFACTS (cite by ID):\n"
            f"{fact_list_with_ids}\n"
        )

    return user_msg


def _validate_anti_repetition(script: dict[str, Any]) -> list[str]:
    blocks = script.get("blocks", [])
    if not blocks:
        return []

    issues: list[str] = []
    texts = [str(b.get("text", "")).lower() for b in blocks if isinstance(b, dict)]

    # Check: same exact number in >2 blocks
    number_counts: dict[str, int] = {}
    for text in texts:
        for match in _NUMBER_RE.findall(text):
            number_counts[match] = number_counts.get(match, 0) + 1
    repeated_numbers = [n for n, c in number_counts.items() if c > 2]
    if repeated_numbers:
        issues.append(
            f"Same number appears in >2 blocks: {', '.join(repeated_numbers)}. "
            "Use each number at most twice."
        )

    # Check: same strong adjective >1 time
    adj_counts: dict[str, int] = {}
    for text in texts:
        words = set(re.findall(r"\b[a-z]+\b", text))
        for word in words & STRONG_ADJECTIVES:
            adj_counts[word] = adj_counts.get(word, 0) + 1
    repeated_adjs = [a for a, c in adj_counts.items() if c > 1]
    if repeated_adjs:
        issues.append(
            f"Strong adjective repeated across blocks: {', '.join(repeated_adjs)}. "
            "Use each strong adjective at most once."
        )

    # Check: same cost/scale framing in >1 block
    cost_blocks = 0
    for text in texts:
        if COST_SCALE_PATTERNS.search(text):
            cost_blocks += 1
    if cost_blocks > 1:
        issues.append(
            f"Cost/scale framing appears in {cost_blocks} blocks. "
            "Use cost/scale framing in at most 1 block."
        )

    return issues


def _parse_script_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    if not text.startswith("{"):
        match = _JSON_OBJECT_RE.search(text)
        if match:
            text = match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM output is not valid JSON script.") from exc


def _validate_script_shape(script: dict[str, Any]) -> None:
    if not isinstance(script, dict):
        raise RuntimeError("Script must be a JSON object.")
    if not isinstance(script.get("language"), str) or not script["language"].strip():
        raise RuntimeError("Script is missing valid 'language'.")
    blocks = script.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise RuntimeError("Script must include non-empty 'blocks'.")
    total_words = 0
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise RuntimeError(f"Block {idx} is not an object.")
        text = block.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"Block {idx} has invalid text.")
        block_words = len(text.split())
        total_words += block_words
        if block_words < 15:
            logger.warning(
                "Block %d is very short (%d words): '%s...'",
                idx, block_words, text[:60],
            )
    if total_words < 80:
        logger.warning(
            "Script total is only %d words (target ≥200). Video will be very short.",
            total_words,
        )
