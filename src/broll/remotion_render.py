"""Remotion render helper for the b-roll Shorts.

Prepares assets (copies into remotion/public/) and invokes the Remotion CLI
to render the BrollShort composition (1080×1920 portrait, paper bg + square
clip montage + animated karaoke captions).
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REMOTION_DIR = PROJECT_ROOT / "remotion"
PUBLIC_DIR   = REMOTION_DIR / "public"

_PAPER_SRC = (
    PROJECT_ROOT
    / "assets" / "videos" / "backgrounds" / "paper"
    / "Gemini_Generated_Image_3flncz3flncz3fln (1).png"
)
_FONT_SRC = PROJECT_ROOT / "assets" / "fonts" / "ChangaOne-Regular.ttf"


def prepare_assets(
    clips_mp4: Path,
    words: list[dict],
    fps: int,
    total_dur: float,
    hook_text: str = "",
) -> dict:
    """Copy assets into remotion/public/ and return the props dict for Remotion."""
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    # Copy the silent square montage
    clips_dest = PUBLIC_DIR / "broll_clips.mp4"
    shutil.copy2(clips_mp4, clips_dest)
    logger.info("Copied clips -> %s", clips_dest)

    # Copy paper background
    paper_dest = PUBLIC_DIR / "paper.png"
    shutil.copy2(_PAPER_SRC, paper_dest)
    logger.info("Copied paper bg -> %s", paper_dest)

    # Copy font
    font_dest = PUBLIC_DIR / "ChangaOne-Regular.ttf"
    shutil.copy2(_FONT_SRC, font_dest)
    logger.info("Copied font -> %s", font_dest)

    duration_in_frames = int(math.ceil(total_dur * fps)) + 6

    props = {
        "clipsFile": "broll_clips.mp4",
        "paperFile": "paper.png",
        "fontFile": "ChangaOne-Regular.ttf",
        "hookText": hook_text,
        "fps": fps,
        "durationInFrames": duration_in_frames,
        "words": [
            {
                "word": w["word"],
                "start": round(float(w["start"]), 3),
                "end": round(float(w["end"]), 3),
            }
            for w in words
        ],
    }

    # Persist props for fast re-renders (skips TTS + Pexels)
    props_path = clips_mp4.parent / "remotion_props.json"
    props_path.write_text(json.dumps(props, indent=2))
    logger.info("Wrote remotion props -> %s", props_path)

    return props


def render(props: dict, out_mp4: Path) -> Path:
    """Run Remotion CLI to render BrollShort composition to out_mp4."""
    props_json = json.dumps(props)
    cmd = [
        "npx", "remotion", "render",
        "src/index.ts",
        "BrollShort",
        str(out_mp4.resolve()),
        "--concurrency=12",
        f"--props={props_json}",
    ]
    logger.info("Remotion render command: %s", " ".join(cmd[:4]) + " BrollShort ...")
    subprocess.run(cmd, cwd=str(REMOTION_DIR), check=True)
    return out_mp4
