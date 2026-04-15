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
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

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


def write_script_from_facts(
    facts: list[dict],
    *,
    language: str = "en-US",
    target_blocks: int = 6,
    style: str | None = None,
    timeout_sec: float = 25.0,
    extra_constraints: list[str] | None = None,
    skill: dict | None = None,
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
        logger.warning("No LLM API key set (GROQ_API_KEY / GEMINI_API_KEY); using direct fallback.")
        return _fallback_script(parsed_facts, language)

    model = os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL

    try:
        script, meta = _generate_via_llm(
            parsed_facts,
            language=language,
            target_blocks=target_blocks,
            style=style,
            model=model,
            timeout_sec=timeout_sec,
            extra_constraints=extra_constraints,
            skill=skill,
        )
    except Exception as exc:
        logger.error("LLM script generation failed: %s", exc)
        return _fallback_script(parsed_facts, language)

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
                model=model,
                timeout_sec=timeout_sec,
                extra_constraints=issues,
                skill=skill,
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


def _fallback_script(facts: list[RankedFact], language: str) -> dict:
    sorted_facts = sorted(facts, key=lambda f: f.score, reverse=True)
    blocks: list[dict[str, str]] = []
    for fact in sorted_facts:
        text = fact.hook_text.strip() or fact.fact_text.strip()
        if text:
            blocks.append({"text": text})
    return {
        "script": {"language": language, "blocks": blocks},
        "meta": {"provider": "fallback", "warnings": ["LLM unavailable; direct assembly used."]},
    }


def _generate_via_llm(
    facts: list[RankedFact],
    *,
    language: str,
    target_blocks: int,
    style: str | None,
    model: str,
    timeout_sec: float,
    extra_constraints: list[str] | None = None,
    skill: dict | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = _build_prompt(
        facts,
        language=language,
        target_blocks=target_blocks,
        style=style,
        extra_constraints=extra_constraints,
        skill=skill,
    )

    from ..llm_client import chat_completion

    content = chat_completion(
        system=(
            "You write short-video narration scripts from fact lists. "
            "Return strict JSON only. Do not invent facts not in the input."
        ),
        user=prompt,
        model=model,
        temperature=0.3,
        timeout=timeout_sec,
    )
    if not content:
        raise RuntimeError("LLM returned empty content.")

    script = _parse_script_json(content)
    _validate_script_shape(script)

    meta: dict[str, Any] = {
        "provider": "groq",
        "model": model,
        "target_blocks": target_blocks,
        "input_facts": len(facts),
        "output_blocks": len(script.get("blocks", [])),
        "retried": extra_constraints is not None,
    }
    return script, meta


def _build_prompt(
    facts: list[RankedFact],
    *,
    language: str,
    target_blocks: int,
    style: str | None,
    extra_constraints: list[str] | None = None,
    skill: dict | None = None,
) -> str:
    normalized_blocks = max(1, int(target_blocks))

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

    # --- Skill-aware style hint ---
    if skill and skill.get("structure", {}).get("tone"):
        style_hint = f"Style hint: {skill['structure']['tone']}."
    elif style:
        style_hint = f"Style hint: {style}."
    else:
        style_hint = "Style hint: fast, curiosity-driven short-form narration with sharp pacing and spoken rhythm."

    constraint_block = ""
    if extra_constraints:
        constraint_block = (
            "\nADDITIONAL CONSTRAINTS (must fix these issues from previous attempt):\n"
            + "\n".join(f"- {c}" for c in extra_constraints)
            + "\n"
        )

    # --- Build skill-specific sections or use defaults ---
    structure = skill.get("structure", {}) if skill else {}

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

    ending_rule = structure.get("ending_rule") or (
        "The LAST sentence must grammatically and semantically CONNECT BACK to the FIRST sentence.\n"
        "- Viewers who let the video replay will hear a continuous, coherent loop.\n"
        "- The ending should feel like a cliffhanger that the hook answers."
    )

    # Language-aware transitions
    is_vietnamese = language.startswith("vi")
    if is_vietnamese:
        transitions = structure.get("transitions_vi") or structure.get("transitions") or [
            "Nhưng đó chưa phải tất cả...", "Điều thú vị hơn là...",
            "Nhưng chờ đã...", "Và đây là phần điên rồ nhất...",
        ]
    else:
        transitions = structure.get("transitions_en") or structure.get("transitions") or [
            "But that's not even the craziest part...", "And here's where it gets insane...",
            "But wait...", "Now here's the thing nobody talks about...",
        ]
    transitions_str = ", ".join(f"'{t}'" for t in transitions)

    # Skill-specific hook examples
    if skill:
        example_hook = skill.get("example_hook_vi" if is_vietnamese else "example_hook_en", "")
    else:
        example_hook = ""

    # Skill prompt injection
    skill_injection = ""
    if skill and skill.get("prompt_injection"):
        skill_injection = f"\nSKILL FORMAT:\n{skill['prompt_injection']}\n"

    prompt = (
        "Build a VIRAL YouTube Shorts narration script from the ranked facts below.\n"
        "Your goal: make viewers STOP scrolling, watch the ENTIRE video, and REPLAY it.\n"
        "\n"
        "WRITING RULES:\n"
        "1) Write ONE SINGLE continuous block of narration. Do NOT split into multiple blocks.\n"
        f"2) Use natural conjunction words and transitions to connect all facts into a flowing story.\n"
        f"   Good transitions: {transitions_str}\n"
        "3) Short, punchy sentences for voice-over. No filler or generic documentary tone.\n"
        "4) Concrete details over broad framing.\n"
        "5) Do not repeat the same number more than twice.\n"
        "6) Do not reuse strong adjectives (staggering, devastating, massive, haunting, "
        "catastrophic, unprecedented, deadly, horrifying).\n"
        "7) Each fact must add new information or escalate stakes.\n"
        "8) Do not invent facts not present in the input.\n"
        "9) The single block must contain ALL facts woven together as one continuous narration.\n"
        "10) Total script must be at least 200 words. A video under 60 seconds is too short.\n"
        "\n"
        f"HOOK (CRITICAL — this decides if viewers stay or scroll away):\n"
        f"- {hook_rule}\n"
        "- NEVER start with character introduction, background context, or plot summary.\n"
        "- NEVER start with 'In [year]...', 'The story of...', 'Once upon a time...', "
        "'Throughout history...', '[Character] is the main character of...'.\n"
        "- NEVER use markdown headings or labels. Write naturally as if speaking to a friend.\n"
    )

    if example_hook:
        prompt += f"- Example hook: '{example_hook}'\n"
    else:
        prompt += (
            "- Good: 'Subaru đã chết chính xác 27 lần. Và lần đau đớn nhất không phải do kẻ thù.'\n"
            "- Good: 'This city was abandoned overnight. 350,000 people just... left.'\n"
        )

    prompt += (
        "- Bad: 'Subaru là nhân vật chính của bộ truyện Re:Zero, một chàng trai trẻ...'\n"
        "- Bad: 'Hôm nay chúng ta cùng tìm hiểu về...'\n"
        "- Bad: '### Chernobyl Disaster' or 'Heading: Five Facts'\n"
        "\n"
        "CONTENT STYLE — LORE, NOT RECAP:\n"
        "- Do NOT summarize plot or introduce characters. The audience ALREADY knows them.\n"
        "- Instead: analyze WHY, reveal HIDDEN details, present THEORIES and DARK SECRETS.\n"
        "- Every sentence should make the viewer think 'I didn't know that!' not 'Yeah, I know.'\n"
        "- Frame as insider knowledge: 'Điều mà 99 phần trăm fan không biết là...'\n"
        "- Use escalating revelations: each fact should be MORE shocking than the last.\n"
        "\n"
        f"ENDING (CRITICAL for algorithm — boosts replay rate over 100 percent):\n"
        f"- {ending_rule}\n"
        "- NEVER end with a dead-end conclusion like '...một trong những nhân vật quan trọng.'\n"
        "- NEVER end with '...changed the world forever', '...for generations to come', "
        "'...a haunting reminder', '...never be the same'.\n"
        "\n"
        f"PACING RULES:\n"
        f"- {pacing_rule}\n"
        "\n"
        "TEXT-TO-SPEECH FORMATTING (CRITICAL):\n"
        "- This text will be read aloud by a TTS engine. Write ONLY natural spoken text.\n"
        "- NEVER use markdown formatting: no ###, no **, no *, no `, no bullet points, no numbered lists.\n"
        "- NEVER use labels like 'Heading 1:', 'Title:', 'Section:', 'Part 1:'.\n"
        "- No emojis, no special characters, no URLs, no percentage signs (write 'phần trăm').\n"
        "- Every word in the output must be something a human would naturally say out loud.\n"
        "\n"
        "IMAGE KEYWORDS:\n"
        "- Output 'image_keywords': a list of 8-12 image search queries.\n"
        "- Each keyword must work as a STANDALONE search on Google/DDG Images.\n"
        "- ALWAYS include the character or franchise name: 'Hakari Kinji JJK' not 'gambling character'.\n"
        "- Mix character shots with scene descriptions: 'Charles Bernard manga JJK' AND 'pachinko machine Japan'.\n"
        "- NEVER use abstract unsearchable concepts: 'cursed energy swirling' is BAD, 'JJK cursed energy purple effect' is GOOD.\n"
        "- Each keyword should be 3-6 words, searchable as-is on image search engines.\n"
        f"{skill_injection}"
        "\n"
        f"{style_hint}\n"
        f"Requested language: {language}\n"
        f"CRITICAL: ALL block text MUST be written in {language}. "
        f"Even if the input facts are in English, you MUST translate and rewrite them in {language}. "
        f"The audience speaks {language}. Do NOT output English text when the requested language is not English.\n"
        f"{constraint_block}"
        "\n"
        "Return strict JSON only with this shape (ONE single block with all narration):\n"
        '{"language": "...", "blocks": [{"role": "narration", "text": "The entire continuous narration here...", '
        '"image_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]}]}\n'
        "IMPORTANT: Output exactly 1 block. Put ALL the narration text in that one block.\n"
        "Include 5-10 image_keywords covering the key visual moments across the whole narration.\n"
        "\n"
        f"INPUT FACTS:\n{json.dumps(facts_payload, ensure_ascii=False, indent=2)}"
    )
    return prompt


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
