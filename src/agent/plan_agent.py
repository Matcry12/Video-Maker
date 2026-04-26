"""Plan agent — extracts topic, language, style, and config from user prompt."""

import json
import logging
import re
from pathlib import Path as _Path
from typing import Optional

from .models import AgentConfig, AgentPlan

_PROMPT_DIR = _Path(__file__).parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")

logger = logging.getLogger(__name__)


def _has_vn_chars(s: str) -> bool:
    # Latin Extended Additional (U+1E00–U+1EFF) covers all Vietnamese diacritics.
    return any(0x1E00 <= ord(c) <= 0x1EFF or 0x00C0 <= ord(c) <= 0x017F for c in s)


def _heuristic_must_cover(prompt: str) -> list[str]:
    """Extract must_cover angles from prompt via simple regex when LLM omits the field."""
    import re as _re
    angles: list[str] = []
    # Patterns: "cover X", "explain X", "compare X", "comparing X", "about X"
    patterns = [
        r"cover(?:ing)?\s+(?:his|her|its|their|the)?\s*([a-zA-Z][a-zA-Z\s]{3,40}?)(?:[,\.]|$)",
        r"explain(?:ing)?\s+(?:his|her|its|their|the)?\s*([a-zA-Z][a-zA-Z\s]{3,40}?)(?:[,\.]|$)",
        r"compar(?:e|ing)\s+(?:with|to|how)?\s*([a-zA-Z][a-zA-Z\s]{3,40}?)(?:[,\.]|$)",
    ]
    for pat in patterns:
        for m in _re.finditer(pat, prompt, _re.IGNORECASE):
            angle = m.group(1).strip().rstrip("., ")
            if 3 <= len(angle) <= 50:
                angles.append(angle)
    return angles[:5]


def plan_from_prompt(
    prompt: str,
    user_config: Optional[AgentConfig] = None,
) -> AgentPlan:
    """Parse a user prompt into a structured AgentPlan.

    Uses a single cheap LLM call (stage="plan", typically 8b) to extract
    structured config. Falls back to regex/heuristic extraction if LLM fails.
    User overrides (AgentConfig) always take precedence over LLM inference.
    """
    try:
        plan = _plan_with_llm(prompt)
    except Exception as exc:
        logger.warning("LLM planning failed (%s). Using heuristic fallback.", exc)
        plan = _plan_heuristic(prompt)

    # Apply user overrides (explicit config always wins)
    if user_config:
        plan = _merge_config(plan, user_config)

    return plan


def _merge_config(plan: AgentPlan, config: AgentConfig) -> AgentPlan:
    """Apply user-provided overrides to the plan. Only set fields override."""
    if config.image_display is not None:
        plan.image_display = config.image_display
    if config.target_blocks is not None:
        plan.target_blocks = min(max(int(config.target_blocks), 3), 10)
    if config.style is not None:
        plan.style = config.style
    if config.voice is not None:
        plan.voice = config.voice
    if config.bgm_mood is not None:
        plan.bgm_mood = config.bgm_mood
    # skill_id stays on user_overrides — script_agent reads it from there
    plan.user_overrides = config
    return plan


def _plan_with_llm(prompt: str) -> AgentPlan:
    """Call the plan LLM. Raises on parse failure so caller can fall back."""
    from .robust_json import extract_json_dict
    from ..llm_client import chat_completion

    template = _load_prompt("plan.txt")
    system = template.replace("{prompt}", prompt)
    raw = chat_completion(
        system=system,
        user="Return the JSON now.",
        stage="plan",
        temperature=0.1,
        timeout=20.0,
    )
    data = extract_json_dict(
        raw,
        required_keys=["topic", "language", "search_queries"],
    )
    if not data:
        raise ValueError(f"plan LLM returned no parseable JSON: {raw[:200]!r}")

    aliases = data.get("entity_aliases") or []
    if not isinstance(aliases, list):
        aliases = [str(aliases)]
    aliases = [str(a).strip() for a in aliases if str(a).strip()][:6]

    raw_queries = data.get("search_queries") or []
    if not isinstance(raw_queries, list):
        raw_queries = [str(raw_queries)]
    queries = [str(q).strip() for q in raw_queries if str(q).strip()]
    if len(queries) > 10:
        logger.info("plan: truncated search_queries from %d to 10", len(queries))
        queries = queries[:10]

    image_display = str(data.get("image_display") or "popup").strip().lower()
    if image_display not in ("popup", "background"):
        image_display = "popup"

    target_blocks_raw = data.get("target_blocks", 6)
    try:
        target_blocks = int(target_blocks_raw)
    except (TypeError, ValueError):
        target_blocks = 6
    target_blocks = min(max(target_blocks, 3), 10)

    domain_prefs = data.get("domain_preferences") or []
    if not isinstance(domain_prefs, list):
        domain_prefs = []
    domain_prefs = [str(d).strip() for d in domain_prefs if str(d).strip()][:6]

    must_cover_raw = data.get("must_cover") or []
    if not isinstance(must_cover_raw, list):
        must_cover_raw = []
    must_cover = [str(a).strip() for a in must_cover_raw if str(a).strip()][:5]

    if not must_cover:
        must_cover = _heuristic_must_cover(prompt)

    entity_cards_raw = data.get("entity_cards") or []
    if not isinstance(entity_cards_raw, list):
        entity_cards_raw = []
    entity_cards = [d for d in entity_cards_raw if isinstance(d, dict)][:10]

    narrative_dynamic = str(data.get("narrative_dynamic") or "").strip()

    return AgentPlan(
        topic=str(data.get("topic") or "").strip(),
        language=str(data.get("language") or "en-US").strip(),
        style=str(data.get("style") or "").strip(),
        topic_category=str(data.get("topic_category") or "").strip(),
        content_type=str(data.get("content_type") or "").strip(),
        mood=str(data.get("mood") or "").strip(),
        hook_strategy=str(data.get("hook_strategy") or "").strip(),
        bgm_mood=str(data.get("bgm_mood") or "").strip(),
        target_blocks=target_blocks,
        image_display=image_display,
        voice=str(data.get("voice") or "").strip(),
        search_queries=queries,
        entity_aliases=aliases,
        domain_preferences=domain_prefs,
        forbidden_entities=[],
        user_prompt=prompt,
        must_cover=must_cover,
        entity_cards=entity_cards,
        narrative_dynamic=narrative_dynamic,
    )


def _plan_heuristic(prompt: str) -> AgentPlan:
    """Fallback: extract topic from prompt using simple heuristics."""
    topic = _extract_topic_heuristic(prompt)
    language = "en-US"

    lower = prompt.lower()
    # Detect Vietnamese: explicit indicators OR Vietnamese Unicode characters in the prompt
    vn_indicators = ["tiếng việt", "vietnamese", "vi-vn", "bằng tiếng việt"]
    if any(ind in lower for ind in vn_indicators) or _has_vn_chars(lower):
        language = "vi-VN"

    style = ""
    style_map = {
        "dramatic": ["dramatic", "intense", "thriller"],
        "educational": ["educational", "explain", "learn", "teach"],
        "cinematic": ["cinematic", "movie", "film"],
        "energetic": ["energetic", "fast", "exciting", "hype"],
    }
    for style_name, keywords in style_map.items():
        if any(kw in lower for kw in keywords):
            style = style_name
            break

    # Default image_display based on style keywords
    bg_keywords = ["scenic", "travel", "nature", "landscape", "mood", "cinematic", "aesthetic", "background"]
    image_display = "background" if any(kw in lower for kw in bg_keywords) else "popup"

    must_cover = _heuristic_must_cover(prompt)

    return AgentPlan(topic=topic, language=language, style=style, image_display=image_display,
                     user_prompt=prompt, must_cover=must_cover)


def _extract_topic_heuristic(prompt: str) -> str:
    """Extract the topic from a prompt string."""
    lower = prompt.strip()
    prefixes = [
        "make a video about ", "create a video about ", "generate a video about ",
        "make a video on ", "create a video on ", "generate a video on ",
        "make video about ", "video about ", "make a short about ",
        "hãy làm video về ", "tạo video về ", "làm video về ",
        "hãy tạo video về ", "làm video ngắn về ",
    ]
    for prefix in prefixes:
        if lower.lower().startswith(prefix):
            lower = lower[len(prefix):]
            break

    # Strip trailing config keywords (not part of the topic)
    import re as _re
    lower = _re.sub(
        r",?\s*(background\s+images?|popup\s+images?|background|popup)\s*$",
        "", lower, flags=_re.IGNORECASE,
    ).strip()

    topic = lower.strip().rstrip(".")
    if not topic:
        topic = prompt.strip()

    return topic
