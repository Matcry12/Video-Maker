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
        _lab_build_card_only,
        _lab_build_single_center_card_only,
        _lab_build_single_center_slide,
        _lab_build_karaoke_ass,
        _lab_mix_bgm,
        _lab_pick_bg_video,
        _lab_pick_bgm,
        _lab_render_clip_video_only,
        _lab_render_with_continuous_bg,
        _lab_split_into_blocks,
        _lab_build_portrait_slide,
    )
    from ..tts import resolve_bg_video_category, resolve_bgm_folder

    settings = load_agent_settings()
    lab_cfg  = settings.get("lab_editor", {}) or {}
    words_per_block = int(lab_cfg.get("words_per_block", 30))
    xfade_dur = float(lab_cfg.get("xfade_duration", 0.20))
    directions = list(lab_cfg.get("transition_directions", ["left", "up", "right", "down"]))
    bgm_volume = float(lab_cfg.get("bgm_volume", 0.15))

    skill_id = str(script.get("skill_id") or "")
    mood = str(script.get("mood") or "")
    bg_category  = resolve_bg_video_category(skill_id)
    bgm_category = resolve_bgm_folder(skill_id, mood, PROJECT_ROOT / "assets" / "audio" / "bgm")
    bg_video = _lab_pick_bg_video(bg_category)
    logger.info(
        "Lab: skill_id=%r  mood=%r  bg_category=%r  bg_video=%s  bgm_category=%r",
        skill_id, mood, bg_category, bg_video.name if bg_video else "none", bgm_category,
    )

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
        if not pp.exists():
            continue
        try:
            from PIL import Image as _TestPIL
            with _TestPIL.open(pp) as _im:
                _ = _im.size  # read header; fails on corrupt/wrong-format files
                _ = _im.mode
            img_paths.append(pp)
        except Exception:
            logger.warning("Skipping unreadable image: %s", pp)
    if not img_paths:
        raise ValueError("run_editor_lab: no valid images in first block")

    tmp_dir = PROJECT_ROOT / "tmp" / "lab_renders" / output_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- TTS ---
    _emit("editor", "Lab: running TTS...", stage="lab_tts")
    from ..tts import TTSEngine
    audio_path = tmp_dir / "narration.wav"
    from ..tts import resolve_tts_voice
    voice = script.get("voice") or first_block.get("voice") or resolve_tts_voice(language="en")
    from ..agent_config import _load_profile
    _tts_cfg = _load_profile().get("tts") or {}
    rate = _tts_cfg.get("default_rate", "18%")
    if not rate.startswith(("+", "-")):
        rate = "+" + rate
    tts = TTSEngine()
    try:
        synth = tts.synthesize(
            text=narration,
            output_path=audio_path,
            voice=voice,
            rate=rate,
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

    import random as _random

    _TARGET_LANDSCAPE = 16 / 9  # ~1.778
    _TARGET_PORTRAIT  =  9 / 16  # ~0.5625

    # Pre-classify by orientation, keeping each image's measured ratio.
    portrait_imgs: list[tuple[float, Path]] = []   # (closeness, path)
    landscape_imgs: list[tuple[float, Path]] = []
    for p in img_paths:
        try:
            with _PILImage.open(p) as im:
                _iw, _ih = im.size
            _ratio = (_iw / _ih) if _ih else 1.0
        except Exception:
            _ratio = 1.0
        if _ratio < 0.9:
            portrait_imgs.append((abs(_ratio - _TARGET_PORTRAIT), p))
        elif _ratio > 1.2:
            landscape_imgs.append((abs(_ratio - _TARGET_LANDSCAPE), p))
        else:
            # Square-ish (0.9–1.2): treat as portrait single card
            portrait_imgs.append((abs(_ratio - _TARGET_PORTRAIT), p))

    # Sort each pool: images closest to the target ratio come first.
    portrait_imgs.sort(key=lambda t: t[0])
    landscape_imgs.sort(key=lambda t: t[0])
    portrait_paths = [p for _, p in portrait_imgs]
    landscape_paths = [p for _, p in landscape_imgs]

    # Random start so first image varies each render.
    p_idx = _random.randint(0, max(0, len(portrait_paths) - 1)) if portrait_paths else 0
    l_idx = _random.randint(0, max(0, len(landscape_paths) - 1)) if landscape_paths else 0

    # Rebind to plain lists for the loop below.
    portrait_imgs = portrait_paths   # type: ignore[assignment]
    landscape_imgs = landscape_paths  # type: ignore[assignment]

    bias_cycle = [0.3, 0.5, 0.7, 0.4, 0.6, 0.3, 0.7, 0.5]
    fg_paths: list[Path] = []
    slide_paths: list[Path] = []
    for i, _block in enumerate(blocks):
        # Block 0: full-frame portrait hook card (character face).
        # Subsequent blocks: ONE image in the upper-middle band, text below it.
        force_portrait = (i == 0)

        if force_portrait:
            pool = portrait_imgs or landscape_imgs or img_paths
            img_path = pool[0]
            single_center = False
        else:
            # One image per block — prefer landscape, then portrait, then any.
            pool = landscape_imgs or portrait_imgs or img_paths
            img_path = pool[l_idx % len(pool)]
            l_idx += 1
            single_center = True

        bias = bias_cycle[i % len(bias_cycle)]
        if bg_video is not None:
            if single_center:
                fg = _lab_build_single_center_card_only(img_path, bias_x=bias)
            else:
                fg = _lab_build_card_only(img_path, bias_x=bias)
            fp = tmp_dir / f"fg_{i}.png"
            fg.save(fp)
            fg_paths.append(fp)
        else:
            if single_center:
                slide = _lab_build_single_center_slide(img_path, bias_x=bias)
            else:
                slide = _lab_build_portrait_slide(img_path, bias_x=bias)
            slide_path = tmp_dir / f"slide_{i}.png"
            slide.save(slide_path, quality=95)
            slide_paths.append(slide_path)

    # --- Timings ---
    block_starts = [float(b[0]["start"]) for b in blocks]
    block_ends   = [float(b[-1]["end"])  for b in blocks]
    # Include inter-block gaps; pad last block by (n-1)*xfade_dur to compensate
    # for time lost to xfade overlaps so -shortest doesn't truncate audio early.
    n_blocks = len(blocks)
    block_durs = [block_starts[i + 1] - block_starts[i] for i in range(n_blocks - 1)]
    block_durs.append(block_ends[-1] - block_starts[-1] + 0.4 + (n_blocks - 1) * xfade_dur)
    audio_start = block_starts[0]

    # --- Global ASS ---
    all_words_global: list[dict] = []
    for block_words in blocks:
        for w in block_words:
            all_words_global.append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"])})
    ass_str = _lab_build_karaoke_ass(all_words_global, audio_start=audio_start)
    global_ass = tmp_dir / "global.ass"
    global_ass.write_text(ass_str, encoding="utf-8")

    # --- Render: bg-video single-pass OR legacy per-clip + xfade ---
    xfade_out = tmp_dir / "xfade.mp4"
    if bg_video is not None:
        _emit("editor", "Lab: rendering with continuous bg + fg xfade...", stage="lab_render_bg")
        _lab_render_with_continuous_bg(
            bg_video=bg_video,
            fg_pngs=fg_paths,
            block_durs=block_durs,
            directions=directions,
            xfade_dur=xfade_dur,
            ass_path=global_ass,
            audio_path=audio_path,
            audio_start=audio_start,
            out=xfade_out,
        )
    else:
        _emit("editor", f"Lab: rendering {len(blocks)} clips...", stage="lab_render")
        clip_paths: list[Path] = []
        for i in range(n_blocks):
            clip_path = tmp_dir / f"clip_{i}.mp4"
            _lab_render_clip_video_only(slide_paths[i], block_durs[i], clip_path)
            clip_paths.append(clip_path)

        _emit("editor", "Lab: applying transitions...", stage="lab_xfade")
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
    # Optional exact-track pin: script.json "bgm_file" overrides mood selection.
    bgm = None
    bgm_file_override = str(script.get("bgm_file") or "").strip()
    if bgm_file_override:
        cand = Path(bgm_file_override)
        if not cand.is_absolute():
            cand = bgm_dir / bgm_file_override
        if cand.exists():
            bgm = cand
            logger.info("Lab: using pinned bgm_file=%s", cand.name)
        else:
            logger.warning("Lab: bgm_file override not found (%s); falling back to mood", cand)
    if bgm is None:
        bgm = _lab_pick_bgm(bgm_dir, category=bgm_category)
    if bgm is not None:
        _emit("editor", f"Lab: mixing BGM ({bgm.name})...", stage="lab_bgm")
        _lab_mix_bgm(xfade_out, bgm, output_path, bgm_vol=bgm_volume)
    else:
        _emit("editor", "Lab: no BGM found, skipping mix", stage="lab_bgm")
        # Just copy xfade_out to output_path
        import shutil as _shutil
        _shutil.copy2(xfade_out, output_path)

    return {"video_path": str(output_path), "audio_path": None}
