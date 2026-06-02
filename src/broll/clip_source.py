"""Stock b-roll clip sourcing — Pexels video search + download + 9:16 fill.

Self-contained; does not modify src/.

Design choices baked in (from researching MoneyPrinterTurbo's failure modes):
- LOOSE resolution filter: prefer portrait renditions, accept anything and
  center-crop-fill to 1080x1920 (MPT's exact-match filter starves the pool and
  its letterbox compositing produces black bars).
- Per-beat clips kept in narration order (no global shuffle).
- Simple on-disk cache keyed by Pexels video id.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

PEXELS_SEARCH = "https://api.pexels.com/videos/search"
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_W, TARGET_H = 1080, 1920


def _api_key() -> str:
    key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "PEXELS_API_KEY not set. Add it to .env (free key at "
            "https://www.pexels.com/api/)."
        )
    return key


def search_clips(query: str, per_page: int = 15, min_duration: int = 3) -> list[dict]:
    """Search Pexels for portrait stock videos matching `query`.

    Returns a list of candidate dicts:
        {"id": int, "duration": float, "best_url": str, "w": int, "h": int,
         "query": str, "image": str, "video_pictures": list[dict]}
    sorted best-first (closest to 9:16, highest resolution). Empty on failure.
    "image" is the Pexels poster thumbnail URL; "video_pictures" is a list of
    {"nr": int, "picture": str} dicts used by the reranker for frame scoring.
    """
    try:
        resp = requests.get(
            PEXELS_SEARCH,
            headers={"Authorization": _api_key()},
            params={
                "query": query,
                "orientation": "portrait",
                "size": "medium",
                "per_page": per_page,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Pexels search failed for %r: %s", query, exc)
        return []

    out: list[dict] = []
    for v in data.get("videos", []):
        dur = float(v.get("duration", 0) or 0)
        if dur < min_duration:
            continue
        best = _pick_file(v.get("video_files", []))
        if not best:
            continue
        out.append({
            "id": v.get("id"),
            "duration": dur,
            "best_url": best["link"],
            "w": best["width"],
            "h": best["height"],
            "query": query,
            "image": v.get("image", ""),
            "video_pictures": v.get("video_pictures", []),
        })

    # Best-first: portrait-ish first, then larger.
    def _score(c: dict) -> tuple:
        is_portrait = 1 if c["h"] >= c["w"] else 0
        return (is_portrait, c["h"] * c["w"])

    out.sort(key=_score, reverse=True)
    return out


def _pick_file(files: list[dict]) -> Optional[dict]:
    """Pick the best downloadable rendition: prefer portrait HD, cap at ~1080 wide."""
    usable = [f for f in files if f.get("link") and f.get("file_type") == "video/mp4"]
    if not usable:
        usable = [f for f in files if f.get("link")]
    if not usable:
        return None

    def _score(f: dict) -> tuple:
        w = int(f.get("width") or 0)
        h = int(f.get("height") or 0)
        is_portrait = 1 if h >= w else 0
        # Prefer renditions close to but not absurdly larger than target height.
        # Penalize huge 4k files (slow download/encode) and tiny sd files.
        height_fit = -abs(h - TARGET_H)
        return (is_portrait, height_fit, h)

    return max(usable, key=_score)


def download_clip(candidate: dict) -> Optional[Path]:
    """Download a candidate clip to the cache (keyed by Pexels id). Returns path."""
    vid = candidate.get("id")
    url = candidate.get("best_url")
    if not url:
        return None
    dest = CACHE_DIR / f"pexels_{vid}.mp4"
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
            tmp.rename(dest)
        return dest
    except Exception as exc:
        logger.warning("Download failed for clip %s: %s", vid, exc)
        return None


def fill_segment(
    src: Path,
    out: Path,
    duration: float,
    start_offset: float = 0.0,
    fps: int = 30,
    layout: str = "fullbleed",
    bg_color: str = "black",
    square: int = 1080,
    blur_sigma: float = 70.0,
) -> Optional[Path]:
    """Cut `duration` seconds from `src` (at native speed) and fit it to the frame.

    layout="fullbleed": scale so the short side covers the 9:16 frame, then
        center-crop the overflow (no letterbox / black bars).
    layout="square": sharp `square`x`square` clip centered over a BLURRED, full-frame
        copy of the same clip (the card-on-blurred-bg look).
    layout="square_color": sharp square centered on a flat `bg_color` canvas
        (blue bands top/bottom).
    """
    duration = max(0.5, float(duration))
    base = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_offset:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-an",
    ]
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)]

    if layout == "square":
        # Blurred full-frame background + sharp centered square on top.
        fc = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_W}:{TARGET_H},gblur=sigma={blur_sigma}[bgb];"
            f"[fg]scale={square}:{square}:force_original_aspect_ratio=increase,"
            f"crop={square}:{square}[sq];"
            f"[bgb][sq]overlay=(W-w)/2:(H-h)/2,fps={fps},setsar=1[v]"
        )
        cmd = base + ["-filter_complex", fc, "-map", "[v]"] + enc
    elif layout == "square_color":
        vf = (
            f"scale={square}:{square}:force_original_aspect_ratio=increase,"
            f"crop={square}:{square},"
            f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:color={bg_color},"
            f"fps={fps},setsar=1"
        )
        cmd = base + ["-vf", vf] + enc
    elif layout == "square_bare":
        # Bare square clip (1080×1080) — no portrait padding.
        # Remotion places this on the paper background.
        vf = (
            f"scale={square}:{square}:force_original_aspect_ratio=increase,"
            f"crop={square}:{square},"
            f"fps={fps},setsar=1"
        )
        cmd = base + ["-vf", vf] + enc
    else:
        vf = (
            f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_W}:{TARGET_H},"
            f"fps={fps},setsar=1"
        )
        cmd = base + ["-vf", vf] + enc
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return out if out.exists() else None
    except subprocess.CalledProcessError as exc:
        logger.warning("fill_segment failed for %s: %s", src, exc.stderr.decode()[:300])
        return None
