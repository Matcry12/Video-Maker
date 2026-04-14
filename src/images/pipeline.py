"""End-to-end image pipeline: search, download, match to script blocks."""

import logging
import re
from pathlib import Path
from typing import Any

from .fetch import download_image, search_images

logger = logging.getLogger(__name__)


def get_images_for_script(
    script: dict[str, Any],
    topic: str,
    topic_category: str = "",
    images_per_block: int = 5,
    **kwargs,  # absorb legacy params like image_mode
) -> dict[int, list[dict]]:
    """Get matched images for each block via per-block search + direct assignment.

    No roundrobin. Each block's images come from its own search query.
    Deduplicates images across blocks by download URL.
    Falls back to broader queries when a block gets 0 images.

    Returns a dict mapping block index to a list of dicts, each with:
        {"path": Path, "keyword": str}
    where keyword is the search keyword used (empty string for topic-level searches).
    """
    blocks = script.get("blocks", [])
    if not blocks:
        return {}

    # Pick sources based on topic category
    sources = _sources_for_topic(topic_category)

    result: dict[int, list[dict]] = {}
    used_urls: set[str] = set()  # Track URLs across blocks to prevent duplicates

    for block_idx, block in enumerate(blocks):
        keywords = block.get("image_keywords", [])

        # Primary: search with topic name directly (most reliable for DDG)
        logger.info("Block %d: searching with topic '%s'", block_idx, topic)
        paths = _search_and_download(topic, sources, images_per_block, used_urls)
        # Convert to dicts with empty keyword (topic-level search)
        block_images = [{"path": p, "keyword": ""} for p in paths]

        # Fallback 1: search each keyword individually if not enough images
        if len(block_images) < images_per_block and keywords:
            for kw in keywords:
                if len(block_images) >= images_per_block:
                    break
                kw_str = str(kw).strip()
                if not kw_str:
                    continue
                logger.info("Block %d: keyword fallback search '%s'", block_idx, kw_str)
                extra = _search_and_download(kw_str, sources, 3, used_urls)
                block_images.extend([{"path": p, "keyword": kw_str} for p in extra])
            block_images = block_images[:images_per_block]

        # Fallback 2: try DDG specifically if not already using it alone
        if not block_images and sources != ["ddg"]:
            logger.info("Block %d: retrying with DDG-only for '%s'", block_idx, topic)
            paths = _search_and_download(topic, ["ddg"], images_per_block, used_urls)
            block_images = [{"path": p, "keyword": ""} for p in paths]

        # Fallback 3: retry without URL dedup — better to reuse images than have gaps
        if not block_images:
            logger.info("Block %d: retrying without dedup (accepting reused images)", block_idx)
            paths = _search_and_download(topic, sources, images_per_block, set())
            block_images = [{"path": p, "keyword": ""} for p in paths]

        if block_images:
            result[block_idx] = block_images
        else:
            logger.warning("Block %d: no images found after all fallbacks.", block_idx)

    total = sum(len(v) for v in result.values())
    logger.info("Downloaded %d images for %d/%d blocks (topic: '%s').", total, len(result), len(blocks), topic)

    return result


def _search_and_download(
    query: str,
    sources: list[str],
    max_images: int,
    used_urls: set[str],
) -> list[Path]:
    """Search for images and download them, skipping already-used URLs."""
    try:
        candidates = search_images(query, sources=sources, per_page=max_images + 5)
    except Exception as exc:
        logger.warning("Image search failed for query '%s': %s", query, exc)
        return []

    paths: list[Path] = []
    for candidate in candidates:
        if len(paths) >= max_images:
            break
        url = candidate.get("url", "")
        if not url or url in used_urls:
            continue
        try:
            path = download_image(url)
        except Exception:
            path = None
        if path and path.exists():
            paths.append(path)
            used_urls.add(url)

    return paths


def _sources_for_topic(topic_category: str) -> list[str]:
    """Pick image sources based on topic category.

    Niche topics (anime, entertainment, trending, biography): DDG only
    Generic topics (history, science, default): DDG + Pixabay + Wikimedia
    """
    niche = {"anime", "entertainment", "trending", "biography", "gaming"}
    if topic_category.lower() in niche:
        return ["ddg"]
    return ["ddg", "pixabay", "wikimedia"]


def _build_image_query(topic: str, block_text: str) -> str:
    """Build an image search query from topic and block text.

    Extracts capitalised words first; falls back to long words.
    Caps the result at 80 characters for clean API results.
    """
    # Prefer capitalised words (likely nouns / named entities)
    words = re.findall(r'\b[A-Z][a-z]{3,}\b', block_text)
    if not words:
        words = [w for w in block_text.split() if len(w) > 5][:3]

    keywords = " ".join(words[:3])
    query = f"{topic} {keywords}".strip()
    return query[:80]
