"""Unified multi-source content fetcher."""

import logging
from typing import Any

from .wikipedia_source import fetch_wikipedia_draft
from .duckduckgo_source import search_duckduckgo
from .crawl4ai_source import crawl_search_results, crawled_to_source_sections

logger = logging.getLogger(__name__)

DEFAULT_MIN_SECTIONS = 4
DEFAULT_MAX_WEB_PAGES = 3


def fetch_topic_content(
    topic: str,
    language: str = "en-US",
    min_sections: int = DEFAULT_MIN_SECTIONS,
    max_web_pages: int = DEFAULT_MAX_WEB_PAGES,
    max_blocks: int = 5,
) -> dict[str, Any]:
    """Fetch content from multiple sources with Wikipedia-first strategy.

    1. Always tries Wikipedia first.
    2. If Wikipedia returns fewer than min_sections, triggers DuckDuckGo + Crawl4AI.
    3. Merges all sources into a unified source_draft dict compatible with
       extract_pipeline.py.

    Never raises — errors from sub-sources are logged and the best available
    result is returned.
    """
    # Step 1: Wikipedia
    wiki_result: dict[str, Any] = {}
    try:
        wiki_result = fetch_wikipedia_draft(topic, language_code=language, max_blocks=max_blocks)
    except Exception as exc:
        logger.warning("Wikipedia fetch failed for '%s': %s", topic, exc)

    source_draft: dict[str, Any] = wiki_result.get("source_draft", {})
    wiki_sections: list[dict[str, Any]] = source_draft.get("sections", [])

    # Step 2: Check if Wikipedia alone is sufficient
    if len(wiki_sections) >= min_sections:
        logger.info(
            "Wikipedia provided %d sections (>= %d min). Skipping web expansion.",
            len(wiki_sections),
            min_sections,
        )
        source_draft["sources_used"] = ["wikipedia"]
        return source_draft

    # Step 3: Expand with DuckDuckGo + Crawl4AI
    logger.info(
        "Wikipedia provided only %d sections (< %d min). Expanding with web sources.",
        len(wiki_sections),
        min_sections,
    )

    search_results: list[dict[str, Any]] = []
    try:
        search_results = search_duckduckgo(topic, language=language)
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for '%s': %s", topic, exc)

    if not search_results:
        logger.warning("DuckDuckGo returned no results for '%s'.", topic)
        source_draft["sources_used"] = ["wikipedia"]
        return source_draft

    crawled: list[dict[str, Any]] = []
    try:
        crawled = crawl_search_results(search_results, max_pages=max_web_pages)
    except Exception as exc:
        logger.warning("Crawl4AI crawling failed for '%s': %s", topic, exc)

    web_sections = crawled_to_source_sections(
        crawled, start_section_id=len(wiki_sections) + 1
    )

    # Step 4: Merge
    merged_sections = wiki_sections + web_sections
    source_draft["sections"] = merged_sections
    source_draft["sources_used"] = ["wikipedia", "duckduckgo", "crawl4ai"]
    source_draft["web_expansion"] = {
        "ddg_results": len(search_results),
        "pages_crawled": len(crawled),
        "web_sections_added": len(web_sections),
    }

    return source_draft
