"""Crawl agent — fetches content, extracts source units, ranks by interest."""

import gc
import logging
import os
from typing import Any, Callable, Optional

from .models import AgentPlan, CrawlResult

logger = logging.getLogger(__name__)


def run_crawl(
    plan: AgentPlan,
    emit: Optional[Callable[[dict], None]] = None,
) -> CrawlResult:
    """Fetch content about the topic, extract units, and rank by interest.

    Merges the old FETCH + EXTRACT + RANK phases into one agent call.
    Never raises — returns CrawlResult with empty facts on failure.
    """
    warnings: list[str] = []

    def _emit(phase: str, message: str, **extra):
        if emit:
            try:
                emit({"phase": phase, "message": message, **extra})
            except Exception:
                pass

    # --- FETCH ---
    _emit("crawl", f"Fetching content about '{plan.topic}'...")

    from ..content_sources.multi_source import fetch_topic_content

    try:
        source_draft = fetch_topic_content(
            topic=plan.topic,
            language=plan.language,
            min_sections=plan.target_blocks,
            max_blocks=12,
            max_web_pages=plan.crawl_max_pages,
        )
    except Exception as exc:
        logger.error("Content fetch failed: %s", exc)
        return CrawlResult(warnings=[f"Content fetch failed: {exc}"])

    sections = source_draft.get("sections", [])
    sources_used = source_draft.get("sources_used", ["wikipedia"])
    logger.info("Fetched %d sections from %s", len(sections), sources_used)

    if not sections:
        return CrawlResult(
            sources_used=sources_used,
            warnings=[f"No content found for topic '{plan.topic}'."],
        )

    # --- EXTRACT ---
    _emit("crawl", "Extracting and scoring content units...")

    from ..content_sources import extract_source_units_from_draft, to_rank_candidates

    extraction = extract_source_units_from_draft(source_draft, keep_top_k=24)
    top_units = extraction.get("top_units", [])

    if not top_units:
        return CrawlResult(
            sources_used=sources_used,
            warnings=["Content extraction produced no usable units."],
        )

    # --- RANK ---
    _emit("crawl", "Ranking content by interest...")

    from ..content_sources import rank_interest_candidates, ranked_items_to_facts

    candidates = to_rank_candidates(top_units)
    ranked_items: list[Any] = []

    interest_model = (
        os.getenv("GROQ_INTEREST_MODEL")
        or os.getenv("GROQ_MODEL")
        or "llama-3.1-8b-instant"
    )

    try:
        ranking_result = rank_interest_candidates(
            candidates, model=interest_model, language=plan.language,
        )
        ranked_items = ranking_result.get("items", [])
    except Exception as exc:
        warnings.append(f"Interest ranking failed: {exc}. Using local scores.")
        logger.warning("Interest ranking failed: %s", exc)

    facts = ranked_items_to_facts(ranked_items, candidates)

    # Fallback: if not enough facts, use raw unit text
    if len(facts) < plan.target_blocks and top_units:
        warnings.append(
            f"Only {len(facts)} ranked facts (need {plan.target_blocks}). "
            "Adding raw units as fallback."
        )
        for u in top_units:
            text = u.get("text", "") if isinstance(u, dict) else ""
            if text and not any(f.get("fact_text") == text for f in facts):
                facts.append({"fact_text": text, "score": 0.0})
            if len(facts) >= plan.target_blocks * 2:
                break

    # Free content pipeline data
    del source_draft, extraction, top_units, candidates, ranked_items
    gc.collect()

    return CrawlResult(
        facts=facts,
        sources_used=sources_used,
        warnings=warnings,
    )
