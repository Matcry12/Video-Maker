"""Wikimedia Commons image source — fallback when DuckDuckGo returns too few images."""

import hashlib
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_TIMEOUT = 8
_API_URL = "https://commons.wikimedia.org/w/api.php"
_MIN_SIZE = 400  # px — reject images smaller than this on the short side
_SKIP_EXTS = {".svg", ".gif", ".webp", ".tiff", ".tif"}
_USER_AGENT = "wikimedia-image-bot/1.0 (video-maker)"

PROJECT_ROOT = Path(__file__).parent.parent.parent
_CACHE_DIR = PROJECT_ROOT / "output" / "wikimedia_cache"


def search_wikimedia_commons(keyword: str, max_images: int = 10) -> list[Path]:
    """Search Wikimedia Commons and download matching images.

    Returns list of local Path objects for downloaded images.
    Never raises — returns [] on any error.
    """
    try:
        return _search_and_download(keyword, max_images)
    except Exception as exc:
        logger.warning("search_wikimedia_commons failed for '%s': %s", keyword, exc)
        return []


def _search_and_download(keyword: str, max_images: int) -> list[Path]:
    """Internal implementation — may raise; caller wraps in try/except."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: search for file titles
    search_params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srnamespace": 6,
        "srsearch": keyword,
        "srlimit": 20,
        "format": "json",
    })
    search_url = f"{_API_URL}?{search_params}"

    try:
        req = urllib.request.Request(search_url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            search_data = json.loads(resp.read().decode(resp.headers.get_content_charset() or "utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("Wikimedia Commons search failed for '%s': %s", keyword, exc)
        return []

    titles = [item["title"] for item in search_data.get("query", {}).get("search", [])]
    if not titles:
        return []

    # Pre-filter titles by extension to avoid unnecessary imageinfo calls
    titles = [t for t in titles if Path(t.split(":")[-1]).suffix.lower() not in _SKIP_EXTS]
    if not titles:
        return []

    # Step 2: fetch imageinfo (url, size, mime) for each title
    info_params = urllib.parse.urlencode({
        "action": "query",
        "titles": "|".join(titles[:20]),
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "format": "json",
    })
    info_url = f"{_API_URL}?{info_params}"

    try:
        req = urllib.request.Request(info_url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            info_data = json.loads(resp.read().decode(resp.headers.get_content_charset() or "utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("Wikimedia Commons imageinfo failed for '%s': %s", keyword, exc)
        return []

    # Collect candidate URLs that pass size and mime filters
    candidates: list[str] = []
    for page in info_data.get("query", {}).get("pages", {}).values():
        infos = page.get("imageinfo", [])
        if not infos:
            continue
        info = infos[0]
        image_url = info.get("url", "")
        if not image_url:
            continue

        # Filter by extension from URL
        ext = Path(image_url.split("?")[0]).suffix.lower()
        if ext in _SKIP_EXTS:
            continue

        # Filter by mime type — only jpeg and png
        mime = info.get("mime", "")
        if mime and mime not in {"image/jpeg", "image/png"}:
            continue

        # Filter by minimum dimension
        width = info.get("width", 0)
        height = info.get("height", 0)
        if width and height and min(width, height) < _MIN_SIZE:
            continue

        candidates.append(image_url)

    if not candidates:
        return []

    # Step 3: download up to max_images, using cache to avoid re-downloading
    paths: list[Path] = []
    for image_url in candidates:
        if len(paths) >= max_images:
            break
        local_path = _download_cached(image_url)
        if local_path is not None:
            paths.append(local_path)

    return paths


def _download_cached(url: str) -> Path | None:
    """Download url to cache dir, reusing existing file if already present.

    Cache filename is SHA-256 of the URL. Returns local Path or None on failure.
    """
    url_hash = hashlib.sha256(url.encode()).hexdigest()

    # Determine extension from URL
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        ext = ".jpg"

    dest = _CACHE_DIR / f"{url_hash}{ext}"

    # Check for cached file with any expected extension
    for candidate_ext in (".jpg", ".jpeg", ".png"):
        candidate = _CACHE_DIR / f"{url_hash}{candidate_ext}"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            content_type = resp.headers.get_content_type() or ""
            if "png" in content_type:
                ext = ".png"
            elif "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"
            dest = _CACHE_DIR / f"{url_hash}{ext}"
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Failed to download Wikimedia image %s: %s", url, exc)
        return None

    if dest.exists() and dest.stat().st_size > 0:
        logger.debug("Downloaded Wikimedia image: %s -> %s", url, dest)
        return dest

    return None
