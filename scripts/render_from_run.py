"""Render a long-form video from an EXISTING run dir (re-uses chapter wavs).

Skips TTS (Chatterbox is slow) by reading `chapter_NN.wav` files written by a
previous `render_novel.py` run, then runs whisper alignment + image pipeline +
`compose_card`. Useful for iterating on the visual editor without redoing audio.

Usage:
    .venv/bin/python scripts/render_from_run.py <run_dir> [output_name]
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("render_from_run")


def _concat_audios(paths: list[Path], out_path: Path, gap_sec: float):
    chunks: list[np.ndarray] = []
    starts: list[float] = []
    sr = None
    cur = 0.0
    for i, p in enumerate(paths):
        data, this_sr = sf.read(str(p), always_2d=False)
        sr = sr or int(this_sr)
        if data.ndim > 1:
            data = data.mean(axis=1)
        starts.append(cur)
        chunks.append(data.astype(np.float32))
        cur += len(data) / sr
        if i < len(paths) - 1:
            chunks.append(np.zeros(int(gap_sec * sr), dtype=np.float32))
            cur += gap_sec
    sf.write(str(out_path), np.concatenate(chunks), sr or 24000)
    return starts, cur


def _align_chapter(wav: Path, text: str) -> list[dict]:
    from faster_whisper import WhisperModel
    from src.whisper_align import _align_to_original
    model = WhisperModel("base", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(str(wav), word_timestamps=True, language="en")
    raw: list[dict] = []
    for seg in segments:
        for w in (seg.words or []):
            tok = str(getattr(w, "word", "") or "").strip()
            if tok:
                raw.append({"word": tok, "start": float(w.start), "end": float(w.end)})
    del model
    return _align_to_original(raw, text) or []


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="output/runs/<name> dir with script.json + chapter_NN.wav")
    ap.add_argument("output_name", nargs="?", default="", help="output mp4 stem")
    ap.add_argument("--mood", default="", help="override BGM mood folder (e.g. edm, hype, somber)")
    ap.add_argument("--bgm-vol", type=float, default=-1.0, help="BGM volume (0.0–1.0); default uses profile")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: {run_dir} not found")
        return 1
    script_path = run_dir / "script.json"
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        return 1

    output_name = args.output_name or f"long_{run_dir.name}_{int(time.time())}"

    script = json.loads(script_path.read_text(encoding="utf-8"))
    chapters_in = script.get("chapters") or []
    if not chapters_in:
        print("Error: no chapters in script.json")
        return 1

    from collections import Counter
    from src.agent_config import load_agent_settings
    from src.agent.image_agent import run_images
    from src.agent.long_editor import compose_card
    from src.agent.models import AgentPlan

    settings = load_agent_settings()
    long_cfg = settings.get("long_form", {}) or {}
    chapter_gap = float(long_cfg.get("chapter_gap_sec", 0.6))
    default_mood = long_cfg.get("default_mood", "")
    topic = script.get("topic") or chapters_in[0].get("title", output_name)
    language = script.get("language", "en-US")

    # 1. Collect existing chapter wavs (intro is ch0 in the script).
    chapter_audios: list[Path] = []
    for i in range(len(chapters_in)):
        wav = run_dir / f"chapter_{i:02d}.wav"
        if not wav.exists():
            print(f"Error: {wav} missing — run render_novel.py first.")
            return 1
        chapter_audios.append(wav)

    # 2. Concat into narration.wav.
    log.info("=== AUDIO CONCAT ===")
    narration_wav = run_dir / "narration.wav"
    chapter_starts, total_dur = _concat_audios(chapter_audios, narration_wav, gap_sec=chapter_gap)
    log.info("Narration: %.2fs (%d chapters, %.2fs gaps)", total_dur, len(chapter_starts), chapter_gap)

    # 3. Whisper align each chapter wav, then offset into global timeline.
    log.info("=== WHISPER ALIGN ===")
    global_words: list[dict] = []
    chapter_durations: list[float] = []
    for i, (wav, ch) in enumerate(zip(chapter_audios, chapters_in)):
        t0 = time.time()
        words = _align_chapter(wav, ch["text"])
        local_dur = max((float(w.get("end", w.get("start", 0.0))) for w in words), default=0.0)
        chapter_durations.append(local_dur)
        offset = chapter_starts[i]
        for w in words:
            global_words.append({
                "word": w["word"],
                "start": float(w["start"]) + offset,
                "end": float(w.get("end", w["start"])) + offset,
            })
        log.info("  ch%d aligned: %d words (%.1fs)", i + 1, len(words), time.time() - t0)
    log.info("Global word timings: %d entries", len(global_words))

    # 4. Image pipeline.
    log.info("=== IMAGE PHASE ===")
    fake_script = {
        "blocks": [
            {
                "role": "narration",
                "text": (ch.get("text") or "")[:600],
                "image_keywords": list(ch.get("image_keywords") or []),
            }
            for ch in chapters_in
        ],
    }
    plan = AgentPlan(
        topic=topic, language=language,
        image_display="background", topic_category="", user_prompt="",
    )

    def _emit(e):
        msg = e.get("message", "")
        if msg:
            log.info("  [%s] %s", e.get("phase", ""), msg)

    image_result = run_images(fake_script, plan, emit=_emit)
    for w in image_result.warnings:
        log.warning("  image: %s", w)
    for idx, paths in image_result.image_map.items():
        log.info("  ch%d → %d images", idx + 1, len(paths))

    # 5. Build chapter timeline.
    chapters_for_editor: list[dict] = []
    moods: list[str] = []
    for i, ch in enumerate(chapters_in):
        start = chapter_starts[i]
        end = start + chapter_durations[i]
        if end <= start + 0.1:
            data, sr = sf.read(str(chapter_audios[i]))
            end = start + len(data) / sr
        block_imgs = image_result.script["blocks"][i].get("image") or []
        img_paths: list[Path] = []
        for p in block_imgs:
            pp = Path(p)
            if not pp.is_absolute():
                pp = PROJECT_ROOT / pp
            if pp.exists():
                img_paths.append(pp)
        if not img_paths:
            raise RuntimeError(f"chapter {i+1} has no usable images")
        mood = (ch.get("mood") or "").strip().lower()
        if mood:
            moods.append(mood)
        chapters_for_editor.append({
            "title": ch.get("title", f"Chapter {i+1}"),
            "start": start, "end": end, "images": img_paths,
        })

    if args.mood:
        chosen_mood = args.mood
        log.info("Mood override: %s", chosen_mood)
    else:
        chosen_mood = Counter(moods).most_common(1)[0][0] if moods else (default_mood or "")
        log.info("Dominant mood: %s", chosen_mood or "(none)")

    # 6. Compose.
    log.info("=== EDITOR PHASE ===")
    output_path = PROJECT_ROOT / "output" / "videos" / f"{output_name}.mp4"
    result = compose_card(
        narration_wav=narration_wav,
        chapters=chapters_for_editor,
        words=global_words,
        output_path=output_path,
        mood=chosen_mood,
        emit=_emit,
        bgm_volume_override=args.bgm_vol if args.bgm_vol >= 0 else None,
    )

    log.info("=== THUMBNAIL + METADATA ===")
    try:
        from src.thumbnail import generate_thumbnail, write_youtube_metadata
        thumb = generate_thumbnail(video_path=output_path, title=topic, run_dir=run_dir)
        log.info("Thumbnail: %s", thumb)
        meta_out = write_youtube_metadata(script_path)
        if meta_out:
            log.info("YT metadata: %s", meta_out)
    except Exception as exc:
        log.warning("Thumbnail/metadata generation failed (non-fatal): %s", exc)

    log.info("=== DONE ===")
    log.info("Video:    %s", result["video_path"])
    log.info("Duration: %.1fs (%.1fmin)", result["duration_sec"], result["duration_sec"] / 60.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
