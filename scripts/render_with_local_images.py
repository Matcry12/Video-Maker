"""Render a script JSON using pre-supplied local images, skipping DDG/Wikimedia search.

Usage:
    .venv/bin/python scripts/render_with_local_images.py <script.json> <output_name> <img1> [img2 ...]

Use this when DDG search returns junk for new/niche topics. Drop curated images
in order and they'll be cycled across the editor's sub-blocks.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def main():
    if len(sys.argv) < 4:
        print("Usage: render_with_local_images.py <script.json> <output_name> <img1> [img2 ...]")
        sys.exit(1)

    script_path = Path(sys.argv[1])
    output_name = sys.argv[2]
    images = [str(Path(p).resolve()) for p in sys.argv[3:]]

    for p in images:
        if not Path(p).exists():
            print(f"Missing image: {p}")
            sys.exit(1)

    script = json.loads(script_path.read_text(encoding="utf-8"))
    blocks = script.get("blocks") or []
    if not blocks:
        print("Script has no blocks.")
        sys.exit(1)

    blocks[0]["image"] = images
    blocks[0]["image_keywords"] = []

    print(f"Script: {len(blocks)} blocks, {len(images)} pre-supplied images")
    for i, p in enumerate(images):
        print(f"  img[{i}] = {Path(p).name}")

    def emit(e):
        msg = e.get("message", "")
        if msg:
            print(f"  [{e.get('phase','')}] {msg}")

    from src.agent_config import load_agent_settings
    settings = load_agent_settings()
    editor_mode = (settings.get("lab_editor", {}) or {}).get("editor_mode", "classic")
    print(f"\n=== EDITOR ({editor_mode}) ===")

    if editor_mode == "lab":
        from src.agent.editor_agent import run_editor_lab
        result = run_editor_lab(script, output_name, emit=emit)
    else:
        from src.agent.editor_agent import run_editor
        result = run_editor(script, output_name, emit=emit)

    print("\n=== DONE ===")
    print(f"Video: {result.get('video_path')}")
    print(f"Audio: {result.get('audio_path')}")


if __name__ == "__main__":
    main()
