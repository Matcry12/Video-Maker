"""SearXNG search adapter — self-hosted meta-search (Bing + Google + DDG aggregated).

Mirrors the duckduckgo_source interface so it can be used as a drop-in
replacement in _phase_search. Falls back gracefully when the local instance
is unreachable.

Configure via SEARXNG_URL env var (default: http://localhost:8080).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
_CACHE_DIR = PROJECT_ROOT / "tmp" / "cache" / "searxng"
_CACHE_TTL_SEC = 24 * 60 * 60
_DEFAULT_MAX_RESULTS = 10
_TIMEOUT_SEC = 10

_SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")


def search_searxng(
    topic: str,
    language: str = "en-US",
    max_results: int = _DEFAULT_MAX_RESULTS,
) -> list[dict[str, Any]]:
    """Search via local SearXNG. Returns [{title, url, snippet}].

    Caches results for 24h. Falls back to an empty list on any error.
    """
    if not topic or not topic.strip():
        return []

    key = _cache_key(topic, language, max_results)
    cached = _load_cached(key)
    if cached is not None:
        logger.debug("SearXNG cache hit for '%s'", topic)
        return cached

    try:
        resp = requests.get(
            f"{_SEARXNG_URL}/search",
            params={
                "q": topic.strip(),
                "format": "json",
                "categories": "general",
                "language": language.split("-")[0] if language else "en",
            },
            timeout=_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("SearXNG search failed for '%s': %s", topic, exc)
        return []

    results: list[dict[str, Any]] = []
    for item in (data.get("results") or []):
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("content") or "").strip()
        if url:
            results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break

    _save_cached(key, results)
    return results


def is_available() -> bool:
    """Quick liveness check — True if SearXNG responds within 3s."""
    try:
        r = requests.get(f"{_SEARXNG_URL}/", timeout=3)
        return r.status_code < 500
    except Exception:
        return False


def _cache_key(topic: str, language: str, max_results: int) -> str:
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
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{key}.json"


def _load_cached(key: str) -> list[dict[str, Any]] | None:
    path = _cache_file_path(key)
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > _CACHE_TTL_SEC:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return None


def _save_cached(key: str, results: list[dict[str, Any]]) -> None:
    path = _cache_file_path(key)
    try:
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.warning("Failed to write SearXNG cache: %s", path)
