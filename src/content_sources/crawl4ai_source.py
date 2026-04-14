"""Web page content extraction using Crawl4AI."""

import asyncio
import logging
import re
from typing import Any

from .text_compressor import compress_for_llm, estimate_tokens

logger = logging.getLogger(__name__)

_CRAWL4AI_AVAILABLE: bool | None = None

_MARKDOWN_SYNTAX_RE = re.compile(
    r"(\*{1,3}|_{1,3}|`{1,3}|#{1,6}\s?|>\s?|\[([^\]]*)\]\([^)]*\)|\!\[[^\]]*\]\([^)]*\))"
)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_WIKIPEDIA_URL_RE = re.compile(r"https?://[a-z]{2,3}\.wikipedia\.org/", re.IGNORECASE)


def _check_crawl4ai() -> bool:
    global _CRAWL4AI_AVAILABLE
    if _CRAWL4AI_AVAILABLE is None:
        try:
            import crawl4ai  # noqa: F401
            _CRAWL4AI_AVAILABLE = True
        except ImportError:
            _CRAWL4AI_AVAILABLE = False
            logger.warning(
                "crawl4ai not installed. Web crawling disabled. "
                "pip install crawl4ai && crawl4ai-setup"
            )
    return _CRAWL4AI_AVAILABLE


def _markdown_to_plain(text: str) -> str:
    """Strip common Markdown syntax to produce plain readable text."""
    plain = _MARKDOWN_SYNTAX_RE.sub(
        lambda m: m.group(2) if m.group(2) is not None else "", text
    )
    plain = _MULTI_NEWLINE_RE.sub("\n\n", plain)
    return plain.strip()


async def _crawl_url_async(url: str, timeout_sec: int = 30) -> dict[str, Any]:
    """Crawl a single URL and return {url, title, text, word_count}."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    content_filter = PruningContentFilter(threshold=0.45, threshold_type="fixed")
    md_generator = DefaultMarkdownGenerator(content_filter=content_filter)

    browser_cfg = BrowserConfig(headless=True, verbose=False)
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        markdown_generator=md_generator,
        page_timeout=timeout_sec * 1000,
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)

    if not result.success:
        logger.warning("Crawl4AI failed for %s: %s", url, result.error_message)
        return {}

    raw_md = ""
    # Try new API first (markdown property returns MarkdownGenerationResult)
    md_obj = getattr(result, "markdown", None)
    if md_obj and hasattr(md_obj, "fit_markdown"):
        raw_md = md_obj.fit_markdown or md_obj.raw_markdown or ""
    elif isinstance(md_obj, str):
        raw_md = md_obj
    if not raw_md:
        raw_md = ""

    plain = _markdown_to_plain(raw_md)
    title = str(result.metadata.get("title") or "").strip() if result.metadata else ""

    return {
        "url": url,
        "title": title,
        "text": plain,
        "word_count": len(plain.split()),
    }


def crawl_url(url: str, timeout_sec: int = 30) -> dict[str, Any]:
    """Synchronous wrapper around _crawl_url_async.

    Returns {url, title, text, word_count} or an empty dict on failure.
    Never raises — all errors are logged and swallowed.
    """
    if not _check_crawl4ai():
        return {}
    try:
        return asyncio.run(_crawl_url_async(url, timeout_sec))
    except Exception as exc:
        logger.warning("Crawl failed for %s: %s", url, exc)
        return {}


async def _crawl_urls_async(
    urls: list[str],
    timeout_sec: int = 30,
) -> list[dict[str, Any]]:
    """Crawl multiple URLs using a single browser instance to save RAM."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    content_filter = PruningContentFilter(threshold=0.45, threshold_type="fixed")
    md_generator = DefaultMarkdownGenerator(content_filter=content_filter)

    browser_cfg = BrowserConfig(headless=True, verbose=False)
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        markdown_generator=md_generator,
        page_timeout=timeout_sec * 1000,
    )

    results: list[dict[str, Any]] = []
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for url in urls:
            try:
                result = await crawler.arun(url=url, config=run_cfg)
            except Exception as exc:
                logger.warning("Crawl4AI failed for %s: %s", url, exc)
                continue

            if not result.success:
                logger.warning("Crawl4AI failed for %s: %s", url, result.error_message)
                continue

            raw_md = ""
            md_obj = getattr(result, "markdown", None)
            if md_obj and hasattr(md_obj, "fit_markdown"):
                raw_md = md_obj.fit_markdown or md_obj.raw_markdown or ""
            elif isinstance(md_obj, str):
                raw_md = md_obj
            if not raw_md:
                raw_md = ""
            plain = _markdown_to_plain(raw_md)
            title = str(result.metadata.get("title") or "").strip() if result.metadata else ""
            results.append({
                "url": url,
                "title": title,
                "text": plain,
                "word_count": len(plain.split()),
            })

    return results


def crawl_search_results(
    search_results: list[dict[str, Any]],
    max_pages: int = 3,
) -> list[dict[str, Any]]:
    """Crawl the top N URLs from DDG search results.

    Uses a single browser instance for all URLs to minimize RAM usage.
    Filters out non-HTTP and Wikipedia URLs (already handled by wikipedia_source).
    Applies text compression and discards pages with fewer than 50 words.
    Returns list of {url, title, text, word_count}.
    """
    if not _check_crawl4ai():
        return []

    urls_to_crawl: list[str] = []
    for item in search_results:
        if len(urls_to_crawl) >= max_pages:
            break
        url = str(item.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        if _WIKIPEDIA_URL_RE.match(url):
            continue
        urls_to_crawl.append(url)

    if not urls_to_crawl:
        return []

    try:
        raw_pages = asyncio.run(_crawl_urls_async(urls_to_crawl))
    except Exception as exc:
        logger.warning("Batch crawl failed: %s", exc)
        return []

    crawled: list[dict[str, Any]] = []
    for page in raw_pages:
        compressed = compress_for_llm(page.get("text", ""), max_chars=4000)
        page["text"] = compressed
        page["word_count"] = len(compressed.split())

        if page["word_count"] < 50:
            logger.debug("Skipping %s — too short after compression (%d words)", page.get("url"), page["word_count"])
            continue

        crawled.append(page)

    return crawled


def crawled_to_source_sections(
    crawled_pages: list[dict[str, Any]],
    start_section_id: int = 100,
) -> list[dict[str, Any]]:
    """Convert crawled pages into source_draft section dicts.

    Long pages (> 3000 chars) are split into multiple sections at paragraph
    boundaries. Each section follows the same schema used by wikipedia_source:
    {section_id, title, text, rank, source_url, token_estimate}
    """
    sections: list[dict[str, Any]] = []
    section_num = start_section_id

    for idx, page in enumerate(crawled_pages):
        url = str(page.get("url") or "")
        title = str(page.get("title") or f"Web Source {idx + 1}").strip()
        text = str(page.get("text") or "").strip()
        if not text:
            continue

        chunks = _split_into_chunks(text, max_chars=3000)
        for chunk_idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            section_title = title if chunk_idx == 0 else f"{title} (cont.)"
            sections.append(
                {
                    "section_id": f"s{section_num:03d}",
                    "title": section_title,
                    "text": chunk,
                    "rank": len(sections) + 1,
                    "source_url": url,
                    "token_estimate": estimate_tokens(chunk),
                }
            )
            section_num += 1

    return sections


def _split_into_chunks(text: str, max_chars: int = 3000) -> list[str]:
    """Split text at paragraph boundaries when it exceeds max_chars."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) + 1 > max_chars and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_len = len(para)
        else:
            current_parts.append(para)
            current_len += len(para) + 1

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# BM25-filtered crawl — topic-aware fit_markdown extraction
# ---------------------------------------------------------------------------

async def _crawl_urls_bm25_async(
    urls: list[str],
    query: str,
    timeout_sec: int = 30,
    bm25_threshold: float = 1.5,
) -> list[dict[str, Any]]:
    """Crawl URLs with BM25ContentFilter for topic-focused fit_markdown.

    Unlike _crawl_urls_async which uses PruningContentFilter,
    this uses BM25ContentFilter with the topic query to produce
    only query-relevant sections of each page.
    """
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.content_filter_strategy import BM25ContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    bm25_filter = BM25ContentFilter(user_query=query, bm25_threshold=bm25_threshold)
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)

    browser_cfg = BrowserConfig(headless=True, verbose=False)
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        markdown_generator=md_generator,
        page_timeout=timeout_sec * 1000,
    )

    results: list[dict[str, Any]] = []
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for url in urls:
            try:
                result = await crawler.arun(url=url, config=run_cfg)
            except Exception as exc:
                logger.warning("BM25 crawl failed for %s: %s", url, exc)
                continue

            if not result.success:
                logger.warning("BM25 crawl failed for %s: %s", url, result.error_message)
                continue

            fit_md = ""
            md_obj = getattr(result, "markdown", None)
            if md_obj and hasattr(md_obj, "fit_markdown"):
                fit_md = md_obj.fit_markdown or ""
            if not fit_md and md_obj and hasattr(md_obj, "raw_markdown"):
                fit_md = md_obj.raw_markdown or ""
            if not fit_md and isinstance(md_obj, str):
                fit_md = md_obj

            plain = _markdown_to_plain(fit_md) if fit_md else ""
            title = str(result.metadata.get("title") or "").strip() if result.metadata else ""
            word_count = len(plain.split()) if plain else 0

            if word_count >= 30:
                results.append({
                    "url": url,
                    "title": title,
                    "text": plain[:4000],
                    "word_count": word_count,
                })

    return results


def crawl_with_bm25(
    urls: list[str],
    query: str,
    max_pages: int = 5,
    bm25_threshold: float = 1.5,
) -> list[dict[str, Any]]:
    """Synchronous wrapper for BM25-filtered crawling.

    Returns list of {url, title, text, word_count} for topic-relevant content.
    """
    if not _check_crawl4ai():
        return []

    urls = urls[:max_pages]
    if not urls:
        return []

    try:
        return asyncio.run(_crawl_urls_bm25_async(urls, query, bm25_threshold=bm25_threshold))
    except Exception as exc:
        logger.warning("BM25 crawl failed: %s", exc)
        return []
