"""
Render a Claude-written script JSON through the image + editor pipeline.

Usage:
    .venv/bin/python scripts/render_script.py <script_json_path> [output_name]

The script JSON must have this format:
{
  "language": "en-US",
  "image_mode": "popup",
  "blocks": [
    {
      "role": "narration",
      "text": "...",
      "image_keywords": ["Keyword One", "Keyword Two Franchise"]
    }
  ]
}
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/render_script.py <script.json> [output_name]")
        sys.exit(1)

    script_path = Path(sys.argv[1])
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        sys.exit(1)

    script = json.loads(script_path.read_text(encoding="utf-8"))
    output_name = sys.argv[2] if len(sys.argv) > 2 else f"claude_video_{int(time.time())}"

    language = script.get("language", "en-US")
    image_mode = script.get("image_mode", "popup")
    topic = script.get("topic", output_name)
    topic_category = script.get("topic_category", "")
    blocks = script.get("blocks", [])

    print(f"Script: {len(blocks)} blocks, language={language}, image_mode={image_mode}")
    for i, b in enumerate(blocks):
        words = len(b.get("text", "").split())
        print(f"  Block {i+1}: {words} words, {len(b.get('image_keywords',[]))} keywords")

    # Build minimal AgentPlan for image pipeline
    from src.agent.models import AgentPlan
    plan = AgentPlan(
        topic=topic,
        language=language,
        image_display=image_mode,
        topic_category=topic_category,
        user_prompt="",
    )

    def emit(e):
        msg = e.get("message", "")
        if msg:
            print(f"  [{e.get('phase','')}] {msg}")

    # === IMAGE PHASE ===
    print("\n=== IMAGE PHASE ===")
    from src.agent.image_agent import run_images
    image_result = run_images(script, plan, emit=emit)
    print(f"Images matched: {len(image_result.image_map)}")
    if image_result.warnings:
        for w in image_result.warnings:
            print(f"  Warning: {w}")

    # === EDITOR PHASE ===
    print("\n=== EDITOR PHASE ===")
    from src.agent_config import load_agent_settings
    settings = load_agent_settings()
    lab_cfg = settings.get("lab_editor", {}) or {}
    editor_mode = lab_cfg.get("editor_mode", "classic")

    if editor_mode == "lab":
        from src.agent.editor_agent import run_editor_lab
        editor_result = run_editor_lab(image_result.script, output_name, emit=emit)
    else:
        from src.agent.editor_agent import run_editor
        editor_result = run_editor(image_result.script, output_name, emit=emit)

    video_path = editor_result.get("video_path", "")
    audio_path = editor_result.get("audio_path", "")

    print(f"\n=== DONE ===")
    print(f"Video: {video_path}")
    print(f"Audio: {audio_path}")


if __name__ == "__main__":
    main()
