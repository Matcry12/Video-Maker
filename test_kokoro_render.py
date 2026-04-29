"""Render an existing script through Kokoro TTS + lab editor end-to-end.

Loads output/runs/<run>/script.json, attaches placeholder images, runs
run_editor_lab. Verifies our Kokoro integration drives a full render.

Run:  ./.venv/bin/python test_kokoro_render.py [run_dir]
"""
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

PROJECT_ROOT = Path(__file__).parent
PLACEHOLDERS = [
    PROJECT_ROOT / "assets/images/overlays/photo_2026-03-29_22-50-08.jpg",
    PROJECT_ROOT / "assets/images/overlays/e8fb756d41c63618535cb93c9720ea86.jpg",
]


def main():
    run_dir_name = sys.argv[1] if len(sys.argv) > 1 else "claude_johnny_vs_giorno_1777312809"
    src_script = PROJECT_ROOT / "output/runs" / run_dir_name / "script.json"
    if not src_script.exists():
        print(f"missing: {src_script}")
        sys.exit(1)

    script = json.loads(src_script.read_text())
    blocks = script.get("blocks", [])
    if not blocks:
        print("script has no blocks")
        sys.exit(1)

    # attach placeholder images to first block (lab editor re-blocks based on word timing)
    blocks[0]["image"] = [str(p) for p in PLACEHOLDERS if p.exists()]
    if not blocks[0]["image"]:
        print("no placeholder images found")
        sys.exit(1)
    print(f"attached {len(blocks[0]['image'])} placeholders")

    # Force Kokoro voice (verifies dispatch + cache + render path)
    script["voice"] = "am_liam"
    print(f"voice: {script['voice']}  text_len: {len(blocks[0]['text'])} chars")

    output_name = f"kokoro_render_{int(time.time())}"

    def emit(e):
        ph = e.get("phase") or e.get("stage", "?")
        msg = e.get("message", "")
        print(f"  [{ph}] {msg}", flush=True)

    from src.agent.editor_agent import run_editor_lab
    t0 = time.monotonic()
    result = run_editor_lab(script, output_name, emit=emit)
    elapsed = time.monotonic() - t0

    print("=" * 60)
    print(f"elapsed: {elapsed:.1f}s")
    print(f"video_path: {result.get('video_path')}")
    print(f"audio_path: {result.get('audio_path')}")
    vp = Path(result.get("video_path") or "")
    if vp.exists():
        print(f"video size: {vp.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
