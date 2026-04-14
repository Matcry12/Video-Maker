"""Skill selector — matches user prompt + plan metadata to the best script skill template.

Uses BM25 (RAGIndex) to match against skill descriptions + trigger keywords,
with bonus scoring for exact content_type and mood matches from the plan agent.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from .models import AgentPlan

logger = logging.getLogger(__name__)

# Module-level cache — skills don't change at runtime
_skills_cache: list[dict[str, Any]] | None = None

_DEFAULT_SKILLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "skills",
)

# Matching thresholds
_CONTENT_TYPE_BONUS = 2.0
_MOOD_BONUS = 1.0
_MIN_SCORE_THRESHOLD = 0.5


def load_skills(skills_dir: str = "") -> list[dict[str, Any]]:
    """Load all skill JSON files from the skills directory.

    Results are cached after the first call.
    """
    global _skills_cache
    if _skills_cache is not None:
        return _skills_cache

    skills_dir = skills_dir or _DEFAULT_SKILLS_DIR
    skills_path = Path(skills_dir)

    if not skills_path.is_dir():
        logger.warning("Skills directory not found: %s", skills_dir)
        _skills_cache = []
        return _skills_cache

    skills: list[dict[str, Any]] = []
    for filepath in sorted(skills_path.glob("*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                skill = json.load(f)
            if "skill_id" not in skill:
                logger.warning("Skipping %s — missing skill_id", filepath.name)
                continue
            skills.append(skill)
        except Exception as exc:
            logger.warning("Failed to load skill %s: %s", filepath.name, exc)

    logger.info("Loaded %d script skills from %s", len(skills), skills_dir)
    _skills_cache = skills
    return _skills_cache


def get_default_skill() -> dict[str, Any]:
    """Return the _default skill (fallback)."""
    skills = load_skills()
    for skill in skills:
        if skill["skill_id"] == "_default":
            return skill
    # If no _default.json exists, return a minimal stub
    return {
        "skill_id": "_default",
        "name": "General",
        "description": "General viral short narration",
        "trigger_keywords": [],
        "content_types": [],
        "moods": [],
        "structure": {},
        "prompt_injection": "",
    }


def select_skill(plan: AgentPlan, prompt: str = "") -> dict[str, Any]:
    """Select the best matching skill for the given plan and prompt.

    Matching strategy:
    1. BM25 score from query against skill description + trigger_keywords
    2. +2.0 bonus if plan.content_type matches skill's content_types
    3. +1.0 bonus if plan.mood matches skill's moods
    4. If best score >= 0.5, use that skill; otherwise use _default

    Returns the skill dict.
    """
    skills = load_skills()
    if not skills:
        return get_default_skill()

    # Filter out _default from candidates (it's the fallback)
    candidates = [s for s in skills if s["skill_id"] != "_default"]
    if not candidates:
        return get_default_skill()

    # Build query from plan fields + prompt
    query_parts = [plan.topic]
    if plan.content_type:
        query_parts.append(plan.content_type)
    if plan.mood:
        query_parts.append(plan.mood)
    if plan.hook_strategy:
        query_parts.append(plan.hook_strategy)
    if plan.style:
        query_parts.append(plan.style)
    if prompt:
        query_parts.append(prompt)
    query = " ".join(query_parts)

    # Build RAG index from skill descriptions
    from ..content_sources.rag_index import RAGIndex

    index = RAGIndex()
    for skill in candidates:
        desc = skill.get("description", "")
        keywords = " ".join(skill.get("trigger_keywords", []))
        index.add({
            "text": f"{desc} {keywords}",
            "skill_id": skill["skill_id"],
        })

    # BM25 query
    results = index.query_with_scores(query, top_k=len(candidates))
    index.clear()

    if not results:
        return get_default_skill()

    # Build skill lookup
    skill_by_id = {s["skill_id"]: s for s in candidates}

    # Score each result with bonuses
    best_skill_id = None
    best_score = -1.0

    for doc, bm25_score in results:
        skill_id = doc["skill_id"]
        skill = skill_by_id.get(skill_id)
        if not skill:
            continue

        final_score = bm25_score

        # Content type bonus
        if plan.content_type and plan.content_type in skill.get("content_types", []):
            final_score += _CONTENT_TYPE_BONUS

        # Mood bonus
        if plan.mood and plan.mood in skill.get("moods", []):
            final_score += _MOOD_BONUS

        if final_score > best_score:
            best_score = final_score
            best_skill_id = skill_id

    if best_score >= _MIN_SCORE_THRESHOLD and best_skill_id:
        selected = skill_by_id[best_skill_id]
        logger.info(
            "Selected skill '%s' (score=%.2f) for topic '%s'",
            selected["skill_id"], best_score, plan.topic,
        )
        return selected

    logger.info(
        "No skill matched above threshold (best=%.2f). Using default.",
        best_score,
    )
    return get_default_skill()


def clear_cache() -> None:
    """Clear the skills cache (for testing)."""
    global _skills_cache
    _skills_cache = None
