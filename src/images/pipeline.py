"""End-to-end image pipeline: search, download, match to script blocks."""

import logging
import random
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .fetch import download_image, search_images
from .wikimedia_source import search_wikimedia_commons
from .dedup import dedup_images
from .anime_filter import is_anime_image

logger = logging.getLogger(__name__)

# Selection quality gates (applied to search candidates before download).
# Domains/engines that consistently yield icon crops, fan edits, or junk.
_BAD_SOURCE_SUBSTRINGS: tuple[str, ...] = (
    "pinimg", "pinterest", "fbcdn", "instagram", "lookaside",
)
_BAD_ENGINE_SUBSTRINGS: tuple[str, ...] = ("artic",)
_MIN_IMAGE_DIM = 500   # drop anything whose shorter side is under this (when known)
_ICON_MAX_DIM = 800    # square images under this are treated as pfp/icon crops


def _passes_quality(candidate: dict[str, Any]) -> bool:
    """Metadata-only pre-download gate: drop junk engines, bad sources,
    tiny images, and square icon/pfp crops. Unknown fields never reject."""
    engine = str(candidate.get("engine") or candidate.get("source") or "")
    if any(b in engine for b in _BAD_ENGINE_SUBSTRINGS):
        return False
    dom = urlparse(candidate.get("url", "")).netloc.replace("www.", "")
    if any(b in dom for b in _BAD_SOURCE_SUBSTRINGS):
        return False
    w = int(candidate.get("width", 0) or 0)
    h = int(candidate.get("height", 0) or 0)
    if w and h:
        if min(w, h) < _MIN_IMAGE_DIM:
            return False
        if w == h and min(w, h) < _ICON_MAX_DIM:
            return False
    return True


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
    # Content-based anime filter for niche/illustration topics
    filter_anime = topic_category in {"anime", "entertainment", "trending"}

    result: dict[int, list[dict]] = {}
    used_urls: set[str] = set()  # Track URLs across blocks to prevent duplicates

    for block_idx, block in enumerate(blocks):
        keywords = block.get("image_keywords", [])
        windows = block.get("windows", [])
        block_images: list[dict] = []

        if windows:
            # Primary: per-window keyword search — gives visual variety across the narration
            per_window = max(2, images_per_block // max(1, len(windows)))
            for win in windows:
                if len(block_images) >= images_per_block:
                    break
                for kw in win.get("image_keywords", [])[:2]:
                    if len(block_images) >= images_per_block:
                        break
                    kw_str = str(kw).strip()
                    if not kw_str:
                        continue
                    logger.info("Block %d window: keyword search '%s'", block_idx, kw_str)
                    extra = _search_and_download(kw_str, sources, per_window, used_urls, filter_anime=filter_anime)
                    block_images.extend([{"path": p, "keyword": kw_str} for p in extra])

            # Fallback: topic search when windows don't fill the quota
            if len(block_images) < images_per_block:
                logger.info("Block %d: topic fallback search '%s'", block_idx, topic)
                paths = _search_and_download(topic, sources, images_per_block - len(block_images), used_urls, filter_anime=filter_anime)
                block_images.extend([{"path": p, "keyword": ""} for p in paths])
        else:
            # No windows — search every keyword (not just top 3) for visual variety
            per_kw = max(2, images_per_block // max(1, len(keywords)))
            for kw in keywords:
                if len(block_images) >= images_per_block:
                    break
                kw_str = str(kw).strip()
                if not kw_str:
                    continue
                logger.info("Block %d: keyword search '%s'", block_idx, kw_str)
                extra = _search_and_download(kw_str, sources, per_kw, used_urls, filter_anime=filter_anime)
                block_images.extend([{"path": p, "keyword": kw_str} for p in extra])

            if len(block_images) < images_per_block:
                logger.info("Block %d: topic fallback search '%s'", block_idx, topic)
                paths = _search_and_download(topic, sources, images_per_block - len(block_images), used_urls, filter_anime=filter_anime)
                block_images.extend([{"path": p, "keyword": ""} for p in paths])

        block_images = block_images[:images_per_block]

        # Fallback 2: Wikimedia Commons — if still fewer than 3 images after keyword search
        if len(block_images) < 3:
            wm_keyword = f"{topic} {keywords[0]}" if keywords else topic
            wm_keyword = wm_keyword.strip()[:80]
            logger.info("Block %d: Wikimedia Commons fallback search '%s'", block_idx, wm_keyword)
            wm_paths = search_wikimedia_commons(wm_keyword, max_images=5)
            block_images.extend([{"path": p, "keyword": wm_keyword} for p in wm_paths])
            block_images = block_images[:images_per_block]

        # Fallback 3: try DDG specifically if not already using it alone
        if not block_images and sources != ["ddg"]:
            logger.info("Block %d: retrying with DDG-only for '%s'", block_idx, topic)
            paths = _search_and_download(topic, ["ddg"], images_per_block, used_urls, filter_anime=filter_anime)
            block_images = [{"path": p, "keyword": ""} for p in paths]

        # Fallback 3: retry without URL dedup or quality gate — better to
        # reuse/accept any image than to leave the block with no visuals.
        if not block_images:
            logger.info("Block %d: retrying without dedup/quality gate (accepting reused images)", block_idx)
            paths = _search_and_download(topic, sources, images_per_block, set(), filter_anime=filter_anime, quality_filter=False)
            block_images = [{"path": p, "keyword": ""} for p in paths]

        if block_images:
            paths_only = [e["path"] for e in block_images if e.get("path")]
            deduped = dedup_images(paths_only)
            deduped_set = set(deduped)
            block_images = [e for e in block_images if e.get("path") in deduped_set]
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
    filter_anime: bool = False,
    quality_filter: bool = True,
) -> list[Path]:
    """Search for images and download them, skipping already-used URLs.

    Selection is RANDOM, not relevance-ranked: candidates passing the
    metadata quality gate (resolution floor, no junk engines/sources, no
    icon crops) are shuffled, then downloaded until ``max_images`` is met.
    This yields variety across renders while excluding low-quality images.

    If ``filter_anime`` is True, downloaded images that fail the anime/real
    classifier are dropped. Over-fetches candidates to absorb drops. Set
    ``quality_filter=False`` for last-resort fallbacks where any image beats
    a gap.
    """
    per_page = max_images * 3 + 5 if filter_anime else max_images + 5
    try:
        candidates = search_images(query, sources=sources, per_page=per_page)
    except Exception as exc:
        logger.warning("Image search failed for query '%s': %s", query, exc)
        return []

    if quality_filter:
        before = len(candidates)
        candidates = [c for c in candidates if _passes_quality(c)]
        if before != len(candidates):
            logger.info("quality gate: query '%s' kept %d/%d candidates", query, len(candidates), before)

    # Random selection (not top-ranked) for variety across renders.
    random.shuffle(candidates)

    paths: list[Path] = []
    dropped = 0
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
            if filter_anime and not is_anime_image(path):
                dropped += 1
                used_urls.add(url)
                continue
            paths.append(path)
            used_urls.add(url)

    if filter_anime and dropped:
        logger.info("anime_filter: query '%s' dropped %d non-anime images, kept %d", query, dropped, len(paths))
    return paths


def _sources_for_topic(topic_category: str) -> list[str]:
    """Pick image sources based on topic category.

    Niche topics (anime, entertainment, trending, biography): SearXNG only
    (aggregates Bing + Google + DDG — best for fan art and official art)
    Generic topics (history, science, default): SearXNG + Pixabay + Wikimedia
    Falls back to ddg if SearXNG is unavailable (handled in fetch.py).
    """
    niche = {"anime", "entertainment", "trending", "biography", "gaming"}
    if topic_category.lower() in niche:
        return ["searxng"]
    return ["searxng", "pixabay", "wikimedia"]


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
