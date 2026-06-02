"""Render a multi-chapter long-form video from a hand- or LLM-authored JSON.

Usage:
    .venv/bin/python scripts/render_novel.py <script.json> [output_name]

Script JSON shape:
{
  "language": "en-US",
  "video_type": "long_form",
  "topic": "Re:Zero Arc 4",
  "voice": "cb:fenrir",
  "chapters": [
    {
      "title": "The Boy Who Fell Through the Sky",
      "text": "<350-450 words of narration prose>",
      "image_keywords": ["Subaru Re:Zero", "Lugnica capital Re:Zero", ...],
      "mood": "tense"
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("render_novel")


def _slug(s: str) -> str:
    return re.sub(r"[^\w]+", "_", s).strip("_").lower()[:50] or "long_video"


def _concat_audios(paths: list[Path], out_path: Path, gap_sec: float) -> tuple[list[float], float]:
    """Concat WAVs in PCM space (no FFmpeg) with `gap_sec` silence between chapters.

    Returns (chapter_start_offsets, total_duration_sec).
    """
    chunks: list[np.ndarray] = []
    starts: list[float] = []
    sr: int | None = None
    cur = 0.0
    for i, p in enumerate(paths):
        data, this_sr = sf.read(str(p), always_2d=False)
        if sr is None:
            sr = int(this_sr)
        elif this_sr != sr:
            raise RuntimeError(f"sample-rate mismatch: {p} is {this_sr}, expected {sr}")
        if data.ndim > 1:
            data = data.mean(axis=1)
        starts.append(cur)
        chunks.append(data.astype(np.float32))
        cur += len(data) / sr
        if i < len(paths) - 1:
            silence = np.zeros(int(gap_sec * sr), dtype=np.float32)
            chunks.append(silence)
            cur += gap_sec
    full = np.concatenate(chunks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), full, sr or 24000)
    return starts, cur


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: render_novel.py <script.json> [output_name]")
        return 1

    script_path = Path(sys.argv[1])
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        return 1

    script = json.loads(script_path.read_text(encoding="utf-8"))
    chapters_in = script.get("chapters") or []
    if not chapters_in:
        print("Error: script.json has no chapters")
        return 1

    output_name = sys.argv[2] if len(sys.argv) > 2 else f"long_{_slug(script.get('topic') or chapters_in[0].get('title', 'video'))}_{int(time.time())}"

    from src.agent_config import _load_profile, load_agent_settings
    from src.agent.image_agent import run_images
    from src.agent.long_editor import compose_card as compose
    from src.agent.models import AgentPlan
    from src.tts import TTSEngine

    settings = load_agent_settings()
    long_cfg = settings.get("long_form", {}) or {}
    profile_voice = long_cfg.get("voice", "cb:fenrir")

    voice = script.get("voice") or profile_voice
    language = script.get("language", "en-US")
    topic = script.get("topic") or chapters_in[0].get("title", output_name)
    chapter_gap = float(long_cfg.get("chapter_gap_sec", 0.6))
    default_mood = long_cfg.get("default_mood", "")

    # Profile rate (matches Shorts behaviour)
    tts_cfg = _load_profile().get("tts") or {}
    rate = tts_cfg.get("default_rate", "+0%")
    if not rate.startswith(("+", "-")):
        rate = "+" + rate

    log.info("Topic: %s | voice=%s | language=%s | chapters=%d", topic, voice, language, len(chapters_in))
    for i, ch in enumerate(chapters_in):
        words = len((ch.get("text") or "").split())
        log.info("  ch%d  %3d words  '%s'", i + 1, words, ch.get("title", ""))

    run_dir = PROJECT_ROOT / "output" / "runs" / output_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")

    # 1. TTS per chapter
    log.info("=== TTS PHASE ===")
    tts = TTSEngine()
    chapter_audios: list[Path] = []
    chapter_words: list[list[dict]] = []
    try:
        for i, ch in enumerate(chapters_in):
            text = (ch.get("text") or "").strip()
            if not text:
                raise RuntimeError(f"chapter {i} has empty text")
            wav_path = run_dir / f"chapter_{i:02d}.wav"
            t0 = time.time()
            result = tts.synthesize(
                text=text,
                output_path=wav_path,
                voice=voice,
                rate=rate,
                alignment_mode="corrected",
            )
            log.info(
                "  ch%d synth: %.2fs audio in %.2fs gen (%d words)",
                i + 1, result["duration"], time.time() - t0, len(result.get("words", [])),
            )
            chapter_audios.append(wav_path)
            chapter_words.append(list(result.get("words") or []))
    finally:
        tts.close()

    # 2. Concat audios + compute global offsets
    log.info("=== AUDIO CONCAT ===")
    narration_wav = run_dir / "narration.wav"
    chapter_starts, total_dur = _concat_audios(chapter_audios, narration_wav, gap_sec=chapter_gap)
    log.info("Narration: %.2fs (%d chapters, %.2fs gaps)", total_dur, len(chapter_starts), chapter_gap)

    # 3. Build global word timeline
    global_words: list[dict] = []
    chapter_durations: list[float] = []
    for i, words in enumerate(chapter_words):
        offset = chapter_starts[i]
        local_dur = 0.0
        for w in words:
            try:
                start = float(w["start"]) + offset
                end = float(w.get("end", w["start"])) + offset
            except Exception:
                continue
            tok = str(w.get("word", "")).strip()
            if not tok:
                continue
            global_words.append({"word": tok, "start": start, "end": end})
            local_dur = max(local_dur, float(w.get("end", w.get("start", 0.0))))
        chapter_durations.append(local_dur)
    log.info("Global word timings: %d entries", len(global_words))

    # 4. Image pipeline — wrap chapters as blocks
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
        topic=topic,
        language=language,
        image_display="background",
        topic_category="",
        user_prompt="",
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

    # 5. Build chapter timeline with images
    chapters_for_editor: list[dict] = []
    moods: list[str] = []
    for i, ch in enumerate(chapters_in):
        start = chapter_starts[i]
        end = start + chapter_durations[i]
        # use chapter audio length if word timings missing
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
            raise RuntimeError(
                f"chapter {i+1} ('{ch.get('title','')}') has no usable images. "
                f"Refine image_keywords and re-run."
            )
        mood = (ch.get("mood") or "").strip().lower()
        if mood:
            moods.append(mood)
        chapters_for_editor.append({
            "title": ch.get("title", f"Chapter {i+1}"),
            "start": start,
            "end": end,
            "images": img_paths,
        })

    chosen_mood = Counter(moods).most_common(1)[0][0] if moods else (default_mood or "")
    log.info("Dominant mood: %s", chosen_mood or "(none → flat BGM dir)")

    # 6. Compose
    log.info("=== EDITOR PHASE ===")
    output_path = PROJECT_ROOT / "output" / "videos" / f"{output_name}.mp4"
    result = compose(
        narration_wav=narration_wav,
        chapters=chapters_for_editor,
        words=global_words,
        output_path=output_path,
        mood=chosen_mood,
        emit=_emit,
    )

    log.info("=== THUMBNAIL + METADATA ===")
    try:
        from src.thumbnail import generate_thumbnail, write_youtube_metadata
        thumb = generate_thumbnail(
            video_path=output_path,
            title=topic,
            run_dir=run_dir,
        )
        log.info("Thumbnail: %s", thumb)
        meta_path = run_dir / "script.json"
        if meta_path.exists():
            meta_out = write_youtube_metadata(meta_path)
            if meta_out:
                log.info("YT metadata: %s", meta_out)
    except Exception as exc:
        log.warning("Thumbnail/metadata generation failed (non-fatal): %s", exc)

    log.info("=== DONE ===")
    log.info("Video:    %s", result["video_path"])
    log.info("Duration: %.1fs (%.1fmin)", result["duration_sec"], result["duration_sec"] / 60.0)
    log.info("Run dir:  %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
