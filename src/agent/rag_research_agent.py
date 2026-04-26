"""Agentic RAG research — wraps lab/research/agent.py for production use.

Drop-in replacement for run_research() when research_mode is "agentic".
Converts the agent's knowledge_doc (## sections) into the facts list format
that the script agent expects.
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
_LAB_RESEARCH = PROJECT_ROOT / "lab" / "research"

# Default skill per topic category when no explicit skill_id is set
_CATEGORY_SKILL_MAP = {
    "anime": "lore_deep_dive",
    "gaming": "lore_deep_dive",
    "entertainment": "explained",
    "music": "explained",
    "celebrity": "dark_secrets",
    "history": "explained",
    "science": "explained",
}
_FALLBACK_SKILL = "_default"


def _resolve_skill(skill_id: str, topic_category: str) -> str:
    if skill_id:
        return skill_id
    return _CATEGORY_SKILL_MAP.get(topic_category, _FALLBACK_SKILL)


def _knowledge_doc_to_facts(knowledge_doc: str) -> tuple[list[dict], list[str]]:
    """Parse ## sections from a knowledge doc into facts list + sources_used."""
    raw_sections = re.split(r'\n##\s+', knowledge_doc.strip())
    facts: list[dict] = []
    sources: list[str] = []

    for i, section in enumerate(raw_sections):
        section = re.sub(r'^##\s+', '', section).strip()
        if not section:
            continue

        lines = section.split('\n', 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        # Collect source URLs from this section
        section_urls = re.findall(r'\[src:\s*(https?://[^\]]+)\]', body)
        for url in section_urls:
            url = url.strip()
            if url not in sources:
                sources.append(url)

        # Strip citation markers and italic formatting for clean fact text
        body = re.sub(r'\s*\[src:[^\]]*\]', '', body).strip()
        body = re.sub(r'\*(.+?)\*', r'\1', body)

        if body:
            # Use the primary source URL for this fact
            source_url = section_urls[0].strip() if section_urls else ""
            facts.append({
                "fact_id": f"F{i+1:03d}",
                "fact_text": f"{title}: {body}",
                "source_url": source_url,
            })

    return facts, sources


def run_agentic_research(
    topic: str,
    search_queries: list[str] | None = None,
    language: str = "en-US",
    skill_id: str = "",
    emit: Optional[Callable[[dict], None]] = None,
    topic_aliases: list[str] | None = None,
    user_prompt: str = "",
    must_cover: list[str] | None = None,
    topic_category: str = "",
) -> dict[str, Any]:
    """Agentic RAG research. Same return shape as run_research().

    Returns:
        {"facts": [...], "sources_used": [...], "warnings": [...], "coverage": str}
    """
    if emit is None:
        emit = lambda _e: None

    warnings: list[str] = []
    resolved_skill = _resolve_skill(skill_id, topic_category)

    # Build prompt — use user_prompt if available, fall back to topic
    prompt = user_prompt or topic

    emit({"phase": "research", "message": f"Agentic research: '{topic}' (skill: {resolved_skill})"})
    logger.info("Agentic research: topic=%r skill=%r provider=gemini", topic, resolved_skill)

    # Load provider/model from profile settings
    from ..agent_config import research_settings
    cfg = research_settings(topic_category=topic_category)
    provider = cfg.get("agentic_provider", "gemini")
    model = cfg.get("agentic_model", "gemma-4-31b-it")

    # Import lab agent — add lab to path if needed
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from lab.research.agent import run as _run_agent
    except ImportError as e:
        warnings.append(f"Agentic agent import failed: {e}. Falling back to classic research.")
        logger.warning("Agentic agent import failed: %s", e)
        from .research_agent import run_research
        return run_research(
            topic=topic,
            search_queries=search_queries,
            language=language,
            skill_id=skill_id,
            emit=emit,
            topic_aliases=topic_aliases,
            user_prompt=user_prompt,
            must_cover=must_cover,
            topic_category=topic_category,
        )

    try:
        result = _run_agent(
            prompt=prompt,
            skill_id=resolved_skill,
            sequential=True,
            model=model,
            provider=provider,
        )
    except Exception as e:
        warnings.append(f"Agentic research failed: {e}. Falling back to classic research.")
        logger.warning("Agentic research failed: %s", e)
        from .research_agent import run_research
        return run_research(
            topic=topic,
            search_queries=search_queries,
            language=language,
            skill_id=skill_id,
            emit=emit,
            topic_aliases=topic_aliases,
            user_prompt=user_prompt,
            must_cover=must_cover,
            topic_category=topic_category,
        )

    knowledge_doc = result.get("knowledge_doc", "")
    metrics = result.get("metrics", {})

    if not knowledge_doc:
        warnings.append("Agentic research returned empty knowledge doc. Falling back.")
        from .research_agent import run_research
        return run_research(
            topic=topic,
            search_queries=search_queries,
            language=language,
            skill_id=skill_id,
            emit=emit,
            topic_aliases=topic_aliases,
            user_prompt=user_prompt,
            must_cover=must_cover,
            topic_category=topic_category,
        )

    facts, sources_used = _knowledge_doc_to_facts(knowledge_doc)

    if not facts:
        warnings.append("Agentic knowledge doc produced no facts. Falling back.")
        from .research_agent import run_research
        return run_research(
            topic=topic,
            search_queries=search_queries,
            language=language,
            skill_id=skill_id,
            emit=emit,
            topic_aliases=topic_aliases,
            user_prompt=user_prompt,
            must_cover=must_cover,
            topic_category=topic_category,
        )

    logger.info(
        "Agentic research done: %d facts, %d sources, %d tool calls, %.1fs",
        len(facts), len(sources_used),
        metrics.get("total_tool_calls", 0),
        metrics.get("elapsed_sec", 0.0),
    )
    emit({
        "phase": "research",
        "message": (
            f"Research complete: {len(facts)} facts from {len(sources_used)} sources "
            f"({metrics.get('total_tool_calls', 0)} searches, {metrics.get('elapsed_sec', 0):.0f}s)"
        ),
    })

    return {
        "facts": facts,
        "sources_used": sources_used,
        "warnings": warnings,
        "coverage": knowledge_doc,
    }
