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
    knowledge_doc: str = "",
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
            knowledge_doc=knowledge_doc,
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
                knowledge_doc=knowledge_doc,
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
    is_vietnamese = language.startswith("vi")
    _topic_sub = video_goal or "the topic"

    # Build skill_prompt_injection from all skill structure fields
    # Skill drives: tone, hook, pacing, ending, style rules — no duplication in base template
    injection_parts: list[str] = []
    if skill:
        structure = skill.get("structure", {})
        tone = structure.get("tone", "") or style or ""
        hook_rule = (structure.get("hook_rule") or "").replace("[topic]", _topic_sub).replace("[year]", "the year")
        pacing_rule = structure.get("pacing_rule") or ""
        ending_rule = structure.get("ending_rule") or ""
        prompt_injection = (skill.get("prompt_injection", "") or "").strip()
        example_hook = (skill.get("example_hook_vi" if is_vietnamese else "example_hook_en", "") or "")
        example_hook = example_hook.replace("[topic]", _topic_sub).replace("[myth]", _topic_sub).replace("[theory]", _topic_sub)

        if tone:
            injection_parts.append(f"STYLE: {tone}")
        if prompt_injection:
            injection_parts.append(prompt_injection)
        # Only add structure fields if not already covered by prompt_injection
        inj_upper = prompt_injection.upper()
        if hook_rule and "HOOK" not in inj_upper:
            hook_block = f"HOOK:\n- {hook_rule}"
            if example_hook:
                hook_block += f"\n- Example: {example_hook}"
            injection_parts.append(hook_block)
        if pacing_rule and "PACING" not in inj_upper:
            injection_parts.append(f"PACING: {pacing_rule}")
        if ending_rule and "ENDING" not in inj_upper:
            injection_parts.append(f"ENDING: {ending_rule}")
    else:
        injection_parts.append(
            "STYLE: fast, curiosity-driven short-form narration with sharp pacing.\n"
            "HOOK: Find the most shocking fact and open with it. First sentence must grab attention immediately.\n"
            "PACING: Alternate short punchy sentences with medium ones. Each fact escalates stakes.\n"
            "ENDING: Last sentence loops back to hook subject with a new revelation."
        )

    skill_prompt_injection = "\n\n".join(injection_parts)

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
            "\nFACTS (cite by ID):\n"
            f"{fact_list_with_ids}"
        )
    else:
        citation_requirement = ""

    return _render_prompt(
        "script_write",
        video_goal=video_goal or "",
        must_cover=must_cover or "",
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
    knowledge_doc: str = "",
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
        knowledge_doc=knowledge_doc,
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
    knowledge_doc: str = "",
) -> str:
    constraint_block = ""
    if extra_constraints:
        constraint_block = (
            "\nADDITIONAL CONSTRAINTS (must fix these issues from previous attempt):\n"
            + "\n".join(f"- {c}" for c in extra_constraints)
            + "\n"
        )

    lang_block = (
        f"Requested language: {language}\n"
        f"CRITICAL: ALL block text MUST be written in {language}. "
        f"Even if the input facts are in English, you MUST translate and rewrite them in {language}. "
        f"The audience speaks {language}. Do NOT output English text when the requested language is not English.\n"
    )

    # Rich research context — strip [src: URL] noise before sending to script LLM
    research_block = ""
    if knowledge_doc:
        clean_doc = re.sub(r'\s*\[src:[^\]]*\]', '', knowledge_doc).strip()
        research_block = f"\nRESEARCH CONTEXT (use for narrative depth and section flow):\n{clean_doc}\n"

    # Citation ID list — always present when facts carry IDs
    facts_with_ids = [f for f in facts if f.fact_id]
    if facts_with_ids:
        fact_list_with_ids = "\n".join(
            f"{f.fact_id}: {f.fact_text[:150]}" for f in facts_with_ids
        )
        cite_block = f"\nFACTS (cite by ID in your script):\n{fact_list_with_ids}\n"
    else:
        # Fallback: no IDs, dump raw facts
        facts_payload = [{"id": i, "fact": f.fact_text} for i, f in enumerate(facts)]
        cite_block = f"\nINPUT FACTS:\n{json.dumps(facts_payload, ensure_ascii=False, indent=2)}\n"

    return f"{lang_block}{constraint_block}{research_block}{cite_block}"


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
