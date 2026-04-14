"""DuckDuckGo search adapter for supplementary content sourcing."""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DDG_CACHE_DIR = PROJECT_ROOT / "tmp" / "cache" / "ddg"
CACHE_TTL_SEC = 24 * 60 * 60
DEFAULT_MAX_RESULTS = 8


def search_duckduckgo(
    topic: str,
    language: str = "en-US",
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[dict[str, Any]]:
    """Search DuckDuckGo for a topic. Returns [{title, url, snippet}].

    Caches results for 24h. Rate-limited with a brief sleep after live requests.
    Falls back to an empty list on any error — never crashes the pipeline.
    """
    if not topic or not topic.strip():
        return []

    key = _cache_key(topic, language, max_results)
    cached = _load_cached(key)
    if cached is not None:
        logger.debug("DDG cache hit for '%s'", topic)
        return cached

    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning(
                "ddgs not installed. DDG search disabled. "
                "pip install ddgs"
            )
            return []

    try:
        ddgs = DDGS(timeout=10)
        region = _language_to_region(language)
        raw = ddgs.text(
            topic.strip(),
            region=region,
            safesearch="moderate",
            max_results=max_results,
        )
        results: list[dict[str, Any]] = []
        for item in (raw or []):
            title = str(item.get("title") or "").strip()
            url = str(item.get("href") or "").strip()
            snippet = str(item.get("body") or "").strip()
            if url:
                results.append({"title": title, "url": url, "snippet": snippet})

        _save_cached(key, results)
        time.sleep(2)
        return results

    except Exception as exc:  # includes RatelimitException, TimeoutException, etc.
        logger.warning("DuckDuckGo search failed for '%s': %s", topic, exc)
        return []


def _language_to_region(language: str) -> str:
    """Map language code like 'vi-VN' to DuckDuckGo region like 'vn-vi'.

    'en-US' -> 'us-en', 'vi-VN' -> 'vn-vi', unknown -> 'wt-wt' (worldwide).
    """
    if not language:
        return "wt-wt"
    parts = str(language).strip().replace("_", "-").split("-")
    if len(parts) == 2:
        lang = parts[0].lower()
        country = parts[1].lower()
        return f"{country}-{lang}"
    return "wt-wt"


def _cache_key(topic: str, language: str, max_results: int) -> str:
    """SHA256-based cache key (first 16 hex chars)."""
    canonical = json.dumps(
        {
            "topic": str(topic or "").strip().casefold(),
            "language": str(language or "").strip().lower(),
            "max_results": int(max_results),
            "version": 1,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _cache_file_path(key: str) -> Path:
    DDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DDG_CACHE_DIR / f"{key}.json"


def _load_cached(key: str) -> list[dict[str, Any]] | None:
    path = _cache_file_path(key)
    if not path.exists():
        return None
    age_sec = time.time() - path.stat().st_mtime
    if age_sec > CACHE_TTL_SEC:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        logger.warning("Failed to read DDG cache file: %s", path)
    return None


def _save_cached(key: str, results: list[dict[str, Any]]) -> None:
    path = _cache_file_path(key)
    try:
        path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        logger.warning("Failed to write DDG cache file: %s", path)
