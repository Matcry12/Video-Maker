"""Plan agent — extracts topic, language, style, and config from user prompt."""

import json
import logging
import os
import re
from typing import Optional

from .models import AgentConfig, AgentPlan

logger = logging.getLogger(__name__)


def plan_from_prompt(
    prompt: str,
    user_config: Optional[AgentConfig] = None,
    model: str = "",
) -> AgentPlan:
    """Parse a user prompt into a structured AgentPlan.

    Uses a single cheap LLM call (8b model) to extract structured config.
    Falls back to regex/heuristic extraction if LLM fails.
    User overrides (AgentConfig) always take precedence over LLM inference.
    """
    model = model or os.getenv("GROQ_MODEL", "") or "llama-3.1-8b-instant"

    try:
        plan = _plan_with_llm(prompt, model)
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
    plan.user_overrides = config
    return plan


def _plan_with_llm(prompt: str, model: str) -> AgentPlan:
    """Use LLM to extract structured plan from prompt."""
    from ..llm_client import chat_completion

    system_prompt = (
        "Extract video generation parameters from the user's request. "
        "Analyze the topic to determine the best content strategy for a viral YouTube Short.\n"
        "Return strict JSON only:\n"
        '{"topic": "...", "language": "en-US or vi-VN", '
        '"style": "dramatic|educational|cinematic|energetic|minimal or empty", '
        '"topic_category": "history|anime|science|biography|trending|entertainment|gaming or empty", '
        '"content_type": "lore|dark_secrets|theory|easter_eggs|comparison|story_time|what_happened_to|timeline|debunked|explained or empty '
        '(lore=deep hidden details, dark_secrets=shocking truths/dark side, theory=fan theories/debates, '
        'easter_eggs=hidden references/foreshadowing, comparison=vs battles, '
        'story_time=narrative storytelling with emotional arc, what_happened_to=nostalgia update/rise and fall, '
        'timeline=chronological evolution/progression, debunked=myth busting/correcting misconceptions, '
        'explained=fast educational breakdown of complex topics)", '
        '"mood": "dark_mystery|epic|emotional|funny|shocking or empty '
        '(the emotional tone of the video)", '
        '"hook_strategy": "lead_with_number|lead_with_question|lead_with_statement|lead_with_contrast or empty '
        '(lead_with_number=shocking statistic, lead_with_question=provocative question, '
        'lead_with_statement=bold controversial claim, lead_with_contrast=Everyone thinks X but actually Y)", '
        '"bgm_mood": "intense|calm|mystery|epic|emotional or empty '
        '(background music mood matching the video tone)", '
        '"target_blocks": 4-8, '
        '"image_display": "popup for storytelling/facts/lists, background for scenic/travel/nature/mood", '
        '"voice": "voice name if specified, else empty", '
        '"search_queries": ["q1","q2","q3","q4","q5","q6"] '
        "(6-8 web search queries targeting DIFFERENT angles: "
        "hidden details, dark facts, fan theories, behind the scenes, "
        "easter eggs, controversies, things you didn't know, psychological analysis. "
        "Use varied wording — each query should find DIFFERENT content. "
        "IMPORTANT: For anime/manga/entertainment/gaming topics, ALWAYS include 3+ English queries "
        "with the original English/romaji title + varied suffixes like 'hidden details', "
        "'dark facts', 'things you didn't know', 'creator interview', 'cut content'. "
        'Example: "Subaru Natsuki Re:Zero dark facts", '
        '"Re:Zero things you didn\'t know", "Re:Zero creator Tappei Nagatsuki interview". '
        "English sources have much more detail for anime/entertainment.)}"
    )

    content = chat_completion(
        system=system_prompt,
        user=prompt,
        model=model,
        temperature=0.1,
        timeout=15.0,
    )

    match = re.search(r'\{.*\}', content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in LLM response: {content[:200]}")

    data = json.loads(match.group())

    # Validate image_display
    raw_display = str(data.get("image_display", "popup")).strip().lower()
    image_display = raw_display if raw_display in ("popup", "background") else "popup"

    # Extract search queries
    raw_queries = data.get("search_queries", [])
    search_queries = [str(q).strip() for q in raw_queries if str(q).strip()][:8]

    return AgentPlan(
        topic=str(data.get("topic", "")).strip() or _extract_topic_heuristic(prompt),
        language=str(data.get("language", "en-US")).strip() or "en-US",
        style=str(data.get("style", "")).strip(),
        topic_category=str(data.get("topic_category", "")).strip(),
        content_type=str(data.get("content_type", "")).strip(),
        mood=str(data.get("mood", "")).strip(),
        hook_strategy=str(data.get("hook_strategy", "")).strip(),
        bgm_mood=str(data.get("bgm_mood", "")).strip(),
        target_blocks=min(max(int(data.get("target_blocks", 6)), 3), 10),
        image_display=image_display,
        voice=str(data.get("voice", "")).strip(),
        search_queries=search_queries,
    )


def _plan_heuristic(prompt: str) -> AgentPlan:
    """Fallback: extract topic from prompt using simple heuristics."""
    topic = _extract_topic_heuristic(prompt)
    language = "en-US"

    lower = prompt.lower()
    # Detect Vietnamese: explicit indicators OR Vietnamese Unicode characters in the prompt
    vn_indicators = ["tiếng việt", "vietnamese", "vi-vn", "bằng tiếng việt"]
    vn_chars = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")
    has_vn_chars = any(c in vn_chars for c in lower)
    if any(ind in lower for ind in vn_indicators) or has_vn_chars:
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

    return AgentPlan(topic=topic, language=language, style=style, image_display=image_display)


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
