"""Image search and download from free APIs."""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
IMAGE_CACHE_DIR = PROJECT_ROOT / "tmp" / "cache" / "images"
SEARCH_CACHE_DIR = PROJECT_ROOT / "tmp" / "cache" / "image_search"
CACHE_TTL_SEC = 7 * 24 * 60 * 60  # 7 days


# ---------------------------------------------------------------------------
# Pixabay
# ---------------------------------------------------------------------------

def search_pixabay(query: str, per_page: int = 10) -> list[dict[str, Any]]:
    """Search Pixabay API. Requires env PIXABAY_API_KEY (free).

    Returns list of dicts with keys: url, preview_url, tags, width, height,
    source, id.  Falls back to empty list if no API key or any error occurs.
    Includes a 1-second sleep between calls to respect rate limits.
    """
    api_key = os.getenv("PIXABAY_API_KEY", "").strip()
    if not api_key:
        logger.debug("PIXABAY_API_KEY not set; skipping Pixabay search.")
        return []

    params = urlencode({
        "key": api_key,
        "q": query,
        "image_type": "photo",
        "safesearch": "true",
        "per_page": min(per_page, 200),
        "min_width": 800,
        "orientation": "horizontal",
    })
    url = f"https://pixabay.com/api/?{params}"

    try:
        time.sleep(1)
        req = Request(url, headers={"User-Agent": "VideoMaker/1.0"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode(resp.headers.get_content_charset() or "utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("Pixabay search failed: %s", exc)
        return []

    results = []
    for hit in data.get("hits", []):
        results.append({
            "url": hit.get("largeImageURL", ""),
            "preview_url": hit.get("previewURL", ""),
            "tags": hit.get("tags", ""),
            "width": hit.get("imageWidth", 0),
            "height": hit.get("imageHeight", 0),
            "source": "pixabay",
            "id": str(hit.get("id", "")),
        })
    return [r for r in results if r["url"]]


# ---------------------------------------------------------------------------
# Wikimedia Commons
# ---------------------------------------------------------------------------

def search_wikimedia(query: str, per_page: int = 10) -> list[dict[str, Any]]:
    """Search Wikimedia Commons. No API key needed.

    Returns list of dicts with keys: url, preview_url, title, source.
    Skips SVG and GIF files.
    """
    _ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    # Step 1: search for file titles
    search_params = urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": 6,
        "srlimit": per_page,
        "format": "json",
    })
    search_url = f"https://commons.wikimedia.org/w/api.php?{search_params}"

    try:
        req = Request(search_url, headers={"User-Agent": "VideoMaker/1.0"})
        with urlopen(req, timeout=15) as resp:
            search_data = json.loads(resp.read().decode(resp.headers.get_content_charset() or "utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("Wikimedia search failed: %s", exc)
        return []

    titles = [item["title"] for item in search_data.get("query", {}).get("search", [])]
    if not titles:
        return []

    # Filter by extension before making imageinfo calls
    titles = [
        t for t in titles
        if Path(t.split(":")[-1]).suffix.lower() in _ALLOWED_EXTS
    ]
    if not titles:
        return []

    # Step 2: get imageinfo for each title
    info_params = urlencode({
        "action": "query",
        "titles": "|".join(titles[:per_page]),
        "prop": "imageinfo",
        "iiprop": "url|size",
        "format": "json",
    })
    info_url = f"https://commons.wikimedia.org/w/api.php?{info_params}"

    try:
        req = Request(info_url, headers={"User-Agent": "VideoMaker/1.0"})
        with urlopen(req, timeout=15) as resp:
            info_data = json.loads(resp.read().decode(resp.headers.get_content_charset() or "utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("Wikimedia imageinfo failed: %s", exc)
        return []

    results = []
    for page in info_data.get("query", {}).get("pages", {}).values():
        title = page.get("title", "")
        infos = page.get("imageinfo", [])
        if not infos:
            continue
        info = infos[0]
        full_url = info.get("url", "")
        if not full_url:
            continue
        ext = Path(full_url.split("?")[0]).suffix.lower()
        if ext not in _ALLOWED_EXTS:
            continue
        results.append({
            "url": full_url,
            "preview_url": full_url,
            "title": title,
            "source": "wikimedia",
        })
    return results


# ---------------------------------------------------------------------------
# DuckDuckGo
# ---------------------------------------------------------------------------

def search_duckduckgo_images(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """DuckDuckGo image search. No API key needed.

    Returns list of dicts with keys: url, preview_url, title, width, height,
    source.  Requires duckduckgo-search package (lazy import).
    """
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            logger.debug("ddgs not installed; skipping DDG image search.")
            return []

    try:
        time.sleep(2)
        with DDGS() as ddgs:
            hits = list(ddgs.images(
                query,
                size="Large",
                type_image="photo",
                max_results=max_results,
            ))
    except Exception as exc:
        logger.warning("DuckDuckGo image search failed: %s", exc)
        return []

    results = []
    for hit in hits:
        url = hit.get("image", "")
        if not url:
            continue
        results.append({
            "url": url,
            "preview_url": hit.get("thumbnail", url),
            "title": hit.get("title", ""),
            "width": hit.get("width", 0),
            "height": hit.get("height", 0),
            "source": "ddg",
        })
    return results


# ---------------------------------------------------------------------------
# Unified search
# ---------------------------------------------------------------------------

def search_images(
    query: str,
    sources: list[str] | None = None,
    per_page: int = 5,
) -> list[dict[str, Any]]:
    """Unified image search across all configured sources.

    Default sources: ["pixabay", "wikimedia", "ddg"].
    Results are deduplicated by URL.  Combined results are cached for 7 days.
    Never raises — any per-source error is caught and logged.
    """
    if sources is None:
        sources = ["pixabay", "wikimedia", "ddg"]

    # Check search cache
    SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(f"{query}|{'|'.join(sources)}|{per_page}".encode()).hexdigest()
    cache_file = SEARCH_CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < CACHE_TTL_SEC:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

    combined: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    _source_fns = {
        "pixabay": lambda: search_pixabay(query, per_page=per_page),
        "wikimedia": lambda: search_wikimedia(query, per_page=per_page),
        "ddg": lambda: search_duckduckgo_images(query, max_results=per_page),
    }

    for source in sources:
        fn = _source_fns.get(source)
        if fn is None:
            logger.warning("Unknown image source: %s", source)
            continue
        try:
            items = fn()
        except Exception as exc:
            logger.warning("Image source '%s' raised unexpectedly: %s", source, exc)
            items = []
        for item in items:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined.append(item)

    # Persist to cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(combined, f)
    except OSError as exc:
        logger.debug("Could not write search cache: %s", exc)

    return combined


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_image(url: str, filename_hint: str = "") -> Path | None:
    """Download image to cache directory.

    Cache key is SHA-256 of the URL.  Skips download if already cached.
    Returns the local Path on success, or None on failure.
    """
    if not url:
        return None

    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    url_hash = hashlib.sha256(url.encode()).hexdigest()

    # Determine extension from URL path
    url_path = url.split("?")[0]
    ext = Path(url_path).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        ext = ".jpg"  # default; will verify from Content-Type below

    cached = IMAGE_CACHE_DIR / f"{url_hash}{ext}"
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    # Also check alternative extensions in case a prior download used a different ext
    for alt_ext in (".jpg", ".jpeg", ".png", ".webp"):
        alt = IMAGE_CACHE_DIR / f"{url_hash}{alt_ext}"
        if alt.exists() and alt.stat().st_size > 0:
            return alt

    try:
        req = Request(url, headers={"User-Agent": "VideoMaker/1.0"})
        with urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get_content_type() or ""
            if "png" in content_type:
                ext = ".png"
            elif "webp" in content_type:
                ext = ".webp"
            elif "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"
            # Stream to disk in chunks to avoid holding full image in memory
            dest = IMAGE_CACHE_DIR / f"{url_hash}{ext}"
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.warning("Failed to download image %s: %s", url, exc)
        return None
    except OSError as exc:
        logger.warning("Failed to write image cache: %s", exc)
        return None

    logger.debug("Downloaded image: %s -> %s", url, dest)
    return dest
