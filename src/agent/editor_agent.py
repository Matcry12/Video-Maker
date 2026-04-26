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


def run_editor_lab(
    script: dict[str, Any],
    output_name: str,
    emit: Optional[Callable[[dict], None]] = None,
) -> dict[str, Any]:
    """Lab render path: portrait/dual-panel slides + karaoke ASS + xfade + BGM.

    Returns {"video_path": str, "audio_path": str | None}.
    Raises on render failure (caller handles).
    """
    import re as _re

    def _emit(phase: str, message: str, **extra):
        if emit:
            try:
                emit({"phase": phase, "message": message, **extra})
            except Exception:
                pass

    from ..agent_config import load_agent_settings
    from ..editor import (
        _lab_apply_xfade,
        _lab_build_dual_panel_slide,
        _lab_build_karaoke_ass,
        _lab_mix_bgm,
        _lab_pick_bgm,
        _lab_render_clip_video_only,
        _lab_split_into_blocks,
        _lab_build_portrait_slide,
    )

    settings = load_agent_settings()
    lab_cfg  = settings.get("lab_editor", {}) or {}
    words_per_block = int(lab_cfg.get("words_per_block", 30))
    xfade_dur = float(lab_cfg.get("xfade_duration", 0.20))
    directions = list(lab_cfg.get("transition_directions", ["left", "up", "right", "down"]))
    bgm_volume = float(lab_cfg.get("bgm_volume", 0.15))

    blocks_in = script.get("blocks") or []
    if not blocks_in:
        raise ValueError("run_editor_lab: script has no blocks")
    first_block = blocks_in[0]
    narration = _re.sub(r"\s+", " ", str(first_block.get("text", ""))).strip()
    if not narration:
        raise ValueError("run_editor_lab: first block has empty text")

    raw_img_paths = first_block.get("image") or []
    if isinstance(raw_img_paths, str):
        raw_img_paths = [raw_img_paths]
    img_paths: list[Path] = []
    for p in raw_img_paths:
        pp = Path(p)
        if not pp.is_absolute():
            pp = PROJECT_ROOT / pp
        if pp.exists():
            img_paths.append(pp)
    if not img_paths:
        raise ValueError("run_editor_lab: no valid images in first block")

    tmp_dir = PROJECT_ROOT / "tmp" / "lab_renders" / output_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- TTS ---
    _emit("editor", "Lab: running TTS...", stage="lab_tts")
    from ..tts import TTSEngine
    audio_path = tmp_dir / "narration.wav"
    voice = script.get("voice") or first_block.get("voice") or "Guy"
    tts = TTSEngine()
    try:
        synth = tts.synthesize(
            text=narration,
            output_path=audio_path,
            voice=voice,
            rate="+2%",
            alignment_mode="corrected",
        )
    finally:
        tts.close()

    # --- Whisper alignment ---
    _emit("editor", "Lab: aligning with Whisper...", stage="lab_align")
    try:
        from faster_whisper import WhisperModel
        from ..whisper_align import _align_to_original

        model = WhisperModel("base", device="cuda", compute_type="float16")
        segments, _info = model.transcribe(str(audio_path), word_timestamps=True, language="en")
        raw_words: list[dict] = []
        for seg in segments:
            for w in (seg.words or []):
                token = str(getattr(w, "word", "") or "").strip()
                if token:
                    raw_words.append({
                        "word": token,
                        "start": round(float(w.start), 3),
                        "end":   round(float(w.end), 3),
                    })
        del model
        words = _align_to_original(raw_words, narration)
    except Exception as exc:
        logger.warning("Lab: Whisper alignment failed (%s); falling back to TTS words", exc)
        words = []

    if not words:
        words = [
            {
                "word": _re.sub(r"\s+", " ", str(item.get("word", "")).strip()),
                "start": float(item.get("start", 0.0)),
                "end":   float(item.get("end", item.get("start", 0.0))),
            }
            for item in synth.get("words", [])
            if str(item.get("word", "")).strip()
        ]
    if not words:
        raise RuntimeError("run_editor_lab: no word timings available")

    # --- Block split ---
    blocks = _lab_split_into_blocks(words, target=words_per_block)
    if not blocks:
        raise RuntimeError("run_editor_lab: no blocks produced")

    # --- Slides ---
    _emit("editor", f"Lab: building slides ({len(blocks)})...", stage="lab_slides")
    try:
        from PIL import Image as _PILImage
    except Exception as exc:
        raise RuntimeError(f"run_editor_lab: PIL required: {exc}")

    slide_paths: list[Path] = []
    bias_cycle = [0.3, 0.5, 0.7, 0.4, 0.6, 0.3, 0.7, 0.5]
    for i, _block in enumerate(blocks):
        img_path = img_paths[i % len(img_paths)]
        try:
            with _PILImage.open(img_path) as im:
                w, h = im.size
            ratio = (w / h) if h else 1.0
        except Exception:
            ratio = 1.0

        if ratio < 0.9:
            slide = _lab_build_portrait_slide(img_path, bias_x=bias_cycle[i % len(bias_cycle)])
        else:
            top_path = img_paths[i % len(img_paths)]
            bot_path = img_paths[(i + 1) % len(img_paths)]
            slide = _lab_build_dual_panel_slide(top_path, bot_path, bias_top=0.35, bias_bottom=0.65)

        slide_path = tmp_dir / f"slide_{i}.png"
        slide.save(slide_path, quality=95)
        slide_paths.append(slide_path)

    # --- Timings ---
    block_starts = [float(b[0]["start"]) for b in blocks]
    block_ends   = [float(b[-1]["end"])  for b in blocks]
    block_durs   = [block_ends[i] - block_starts[i] for i in range(len(blocks))]
    block_durs[-1] += 0.4  # tail on last block
    audio_start = block_starts[0]

    # --- Render video-only clips ---
    _emit("editor", f"Lab: rendering {len(blocks)} clips...", stage="lab_render")
    clip_paths: list[Path] = []
    for i in range(len(blocks)):
        clip_path = tmp_dir / f"clip_{i}.mp4"
        _lab_render_clip_video_only(slide_paths[i], block_durs[i], clip_path)
        clip_paths.append(clip_path)

    # --- Global ASS ---
    all_words_global: list[dict] = []
    for block_words in blocks:
        for w in block_words:
            all_words_global.append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"])})
    ass_str = _lab_build_karaoke_ass(all_words_global, audio_start=audio_start)
    global_ass = tmp_dir / "global.ass"
    global_ass.write_text(ass_str, encoding="utf-8")

    # --- Apply xfade + ASS + audio ---
    _emit("editor", "Lab: applying transitions...", stage="lab_xfade")
    xfade_out = tmp_dir / "xfade.mp4"
    _lab_apply_xfade(
        clip_paths=clip_paths,
        block_durs=block_durs,
        directions=directions,
        xfade_dur=xfade_dur,
        ass_path=global_ass,
        audio_path=audio_path,
        audio_start=audio_start,
        out=xfade_out,
    )

    # --- BGM mix ---
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}.mp4"

    bgm_dir = PROJECT_ROOT / "assets" / "audio" / "bgm"
    bgm = _lab_pick_bgm(bgm_dir)
    if bgm is not None:
        _emit("editor", f"Lab: mixing BGM ({bgm.name})...", stage="lab_bgm")
        _lab_mix_bgm(xfade_out, bgm, output_path, bgm_vol=bgm_volume)
    else:
        _emit("editor", "Lab: no BGM found, skipping mix", stage="lab_bgm")
        # Just copy xfade_out to output_path
        import shutil as _shutil
        _shutil.copy2(xfade_out, output_path)

    return {"video_path": str(output_path), "audio_path": None}
