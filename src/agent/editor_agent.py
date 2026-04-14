"""Editor agent — renders the final video from a prepared script."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_editor(
    script: dict[str, Any],
    output_name: str,
    emit: Optional[Callable[[dict], None]] = None,
) -> dict[str, Any]:
    """Render a video from a fully prepared script (with images attached).

    Returns {"video_path": str, "audio_path": str | None}.
    Raises on render failure (caller handles).
    """
    def _emit(phase: str, message: str, **extra):
        if emit:
            try:
                emit({"phase": phase, "message": message, **extra})
            except Exception:
                pass

    _emit("editor", "Rendering video...", stage="rendering")

    from ..manager import VideoManager

    script_path = (
        PROJECT_ROOT
        / "tmp"
        / "agent_scripts"
        / f"{output_name}_{int(time.time() * 1000)}.json"
    )
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(json.dumps(script, ensure_ascii=False, indent=2))

    manager = None
    result: dict[str, Any] = {"video_path": None, "audio_path": None}

    try:
        manager = VideoManager()

        def on_render_progress(event: dict):
            _emit(
                "editor",
                event.get("message", "Rendering..."),
                stage=event.get("stage", "rendering"),
                current_block=event.get("current_block"),
                total_blocks=event.get("total_blocks"),
            )

        video_path = manager.process_script(
            script_path, output_name, progress_callback=on_render_progress,
        )
        result["video_path"] = str(video_path)

        audio_path = video_path.with_suffix(".mp3")
        if audio_path.exists():
            result["audio_path"] = str(audio_path)
    finally:
        if manager:
            manager.close()
        # Keep script JSON for tracking — don't delete

    return result
