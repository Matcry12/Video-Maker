"""Render a two-character dialogue video (Marcus & Maya) from a conversation script.

Usage:
    .venv/bin/python scripts/render_conversation.py <conversation_script.json> [output_name]

Script JSON shape:
{
  "topic": "...",
  "language": "en-US",
  "sections": [
    {
      "title": "...",
      "image_keywords": [...],
      "mood": "...",
      "turns": [
        {"speaker": "maya"|"marcus", "line": "...", "exag": 0.6}
      ]
    }
  ],
  "youtube": {...}
}

Per-turn Kokoro TTS (am_fenrir for Marcus, af_heart for Maya), then
section-wise concat, word timeline from synthesis (no Whisper needed),
image pipeline, and compose.
"""

from __future__ import annotations

import json
import logging
import re
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
log = logging.getLogger("render_conversation")


VOICE_MAP = {"marcus": "am_fenrir", "maya": "af_heart"}
TURN_GAP_SEC = 0.2   # silence between turns from different speakers
SAME_GAP_SEC = 0.05  # silence between same-speaker consecutive turns


def _slug(s: str) -> str:
    return re.sub(r"[^\w]+", "_", s).strip("_").lower()[:50] or "conversation"


def _concat_turns_to_section(turn_paths: list[Path], speakers: list[str], out_path: Path) -> float:
    """Concat turn wavs into one section wav with speaker-aware silence gaps."""
    chunks: list[np.ndarray] = []
    sr: int | None = None
    for i, p in enumerate(turn_paths):
        data, this_sr = sf.read(str(p), always_2d=False)
        if sr is None:
            sr = int(this_sr)
        elif this_sr != sr:
            raise RuntimeError(f"sample-rate mismatch: {p} is {this_sr}, expected {sr}")
        if data.ndim > 1:
            data = data.mean(axis=1)
        chunks.append(data.astype(np.float32))
        if i < len(turn_paths) - 1:
            gap = SAME_GAP_SEC if speakers[i] == speakers[i + 1] else TURN_GAP_SEC
            chunks.append(np.zeros(int(gap * sr), dtype=np.float32))
    full = np.concatenate(chunks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), full, sr or 24000)
    return len(full) / (sr or 24000)


def _concat_audios(paths: list[Path], out_path: Path, gap_sec: float) -> tuple[list[float], float]:
    """Concat section WAVs in PCM space with `gap_sec` silence between sections."""
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
            chunks.append(np.zeros(int(gap_sec * sr), dtype=np.float32))
            cur += gap_sec
    full = np.concatenate(chunks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), full, sr or 24000)
    return starts, cur


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: render_conversation.py <conversation_script.json> [output_name]")
        return 1

    script_path = Path(sys.argv[1])
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        return 1

    conversation_script = json.loads(script_path.read_text(encoding="utf-8"))
    sections = conversation_script.get("sections") or []
    if not sections:
        print("Error: conversation script has no sections")
        return 1
    for si, section in enumerate(sections):
        if not (section.get("turns") or []):
            print(f"Error: section {si} ('{section.get('title','')}') has no turns")
            return 1

    topic = conversation_script.get("topic") or sections[0].get("title", "conversation")
    language = conversation_script.get("language", "en-US")

    output_name = sys.argv[2] if len(sys.argv) > 2 else f"conv_{_slug(topic)}_{int(time.time())}"

    from src.agent_config import load_agent_settings, _load_profile
    from src.agent.image_agent import run_images
    from src.agent.long_editor import compose_card
    from src.agent.models import AgentPlan
    from src.tts import TTSEngine

    settings = load_agent_settings()
    long_cfg = settings.get("long_form", {}) or {}
    chapter_gap = float(long_cfg.get("chapter_gap_sec", 0.6))
    default_mood = long_cfg.get("default_mood", "")

    tts_cfg = _load_profile().get("tts") or {}
    rate = tts_cfg.get("default_rate", "+0%")
    if not rate.startswith(("+", "-")):
        rate = "+" + rate

    total_turns = sum(len(s.get("turns") or []) for s in sections)
    log.info(
        "Topic: %s | language=%s | sections=%d | turns=%d | tts=kokoro | rate=%s",
        topic, language, len(sections), total_turns, rate,
    )

    run_dir = PROJECT_ROOT / "output" / "runs" / output_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "script.json").write_text(
        json.dumps(conversation_script, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    saved_script_path = run_dir / "script.json"

    # === PHASE 1: TTS PER TURN (Kokoro — word timings returned directly) ===
    log.info("=== TTS PHASE (per-turn Kokoro) ===")
    tts = TTSEngine()

    # section_turn_data[si][ti] = (wav_path, speaker, duration, words)
    section_turn_data: list[list[tuple[Path, str, float, list[dict]]]] = []
    try:
        for si, section in enumerate(sections):
            turns = section.get("turns") or []
            turn_data: list[tuple[Path, str, float, list[dict]]] = []
            for ti, turn in enumerate(turns):
                speaker = str(turn.get("speaker", "")).strip().lower()
                if speaker not in VOICE_MAP:
                    raise RuntimeError(
                        f"section {si} turn {ti}: unknown speaker {speaker!r} (expected {list(VOICE_MAP)})"
                    )
                line = str(turn.get("line", "")).strip()
                if not line:
                    raise RuntimeError(f"section {si} turn {ti}: empty line")
                voice = VOICE_MAP[speaker]
                out_wav = run_dir / f"turn_{si:02d}_{ti:02d}_{speaker}.wav"
                t0 = time.time()
                result = tts.synthesize(
                    text=line,
                    output_path=out_wav,
                    voice=voice,
                    rate=rate,
                    alignment_mode="corrected",
                )
                duration = float(result.get("duration", 0.0))
                words = list(result.get("words") or [])
                log.info(
                    "[%d/%d] [%d/%d] %s: duration=%.2fs gen=%.2fs words=%d",
                    si + 1, len(sections), ti + 1, len(turns),
                    speaker.upper(), duration, time.time() - t0, len(words),
                )
                turn_data.append((out_wav, speaker, duration, words))
            section_turn_data.append(turn_data)
    finally:
        tts.close()

    # === PHASE 2: BUILD SECTION WAVs + NARRATION.WAV ===
    log.info("=== AUDIO CONCAT (sections + narration) ===")
    section_wavs: list[Path] = []
    section_turn_wavs: list[list[Path]] = []
    section_turn_speakers: list[list[str]] = []
    for si, turn_data in enumerate(section_turn_data):
        section_turn_wavs.append([td[0] for td in turn_data])
        section_turn_speakers.append([td[1] for td in turn_data])
        sec_wav = run_dir / f"section_{si:02d}.wav"
        sec_dur = _concat_turns_to_section(
            section_turn_wavs[si], section_turn_speakers[si], sec_wav
        )
        log.info("  section %d: %.2fs (%d turns)", si + 1, sec_dur, len(turn_data))
        section_wavs.append(sec_wav)

    narration_wav = run_dir / "narration.wav"
    section_starts, total_dur = _concat_audios(section_wavs, narration_wav, gap_sec=chapter_gap)
    log.info(
        "Narration: %.2fs (%d sections, %.2fs gaps) -> %s",
        total_dur, len(section_starts), chapter_gap, narration_wav,
    )

    # === PHASE 3: BUILD GLOBAL WORD TIMELINE (from Kokoro synthesis, no Whisper needed) ===
    log.info("=== WORD TIMELINE (from Kokoro synthesis) ===")
    global_words: list[dict] = []
    section_durations: list[float] = []
    for si, turn_data in enumerate(section_turn_data):
        section_offset = section_starts[si]
        turn_cursor = 0.0
        section_word_count = 0
        for td_idx, (wav_path, speaker, duration, words) in enumerate(turn_data):
            for w in words:
                try:
                    start = float(w["start"]) + turn_cursor + section_offset
                    end = float(w.get("end", w["start"])) + turn_cursor + section_offset
                except Exception:
                    continue
                tok = str(w.get("word", "")).strip()
                if tok:
                    global_words.append({"word": tok, "start": start, "end": end})
                    section_word_count += 1
            turn_cursor += duration
            if td_idx < len(turn_data) - 1:
                next_speaker = turn_data[td_idx + 1][1]
                turn_cursor += TURN_GAP_SEC if speaker != next_speaker else SAME_GAP_SEC
        section_durations.append(turn_cursor)
        log.info("  sec%d: %d words, %.2fs", si + 1, section_word_count, turn_cursor)
    log.info("Global word timings: %d entries", len(global_words))

    # === PHASE 4: IMAGE PIPELINE ===
    log.info("=== IMAGE PHASE ===")
    fake_script = {
        "blocks": [
            {
                "role": "narration",
                "text": " ".join(
                    str(t.get("line", "")).strip() for t in section.get("turns") or []
                )[:600],
                "image_keywords": list(section.get("image_keywords") or []),
            }
            for section in sections
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
        log.info("  sec%d → %d images", idx + 1, len(paths))

    # === PHASE 5: BUILD CHAPTERS FOR EDITOR ===
    chapters_for_editor: list[dict] = []
    moods: list[str] = []
    for si, section in enumerate(sections):
        start = section_starts[si]
        end = start + section_durations[si]
        if end <= start + 0.1:
            data, sr = sf.read(str(section_wavs[si]))
            end = start + len(data) / sr
        block_imgs = image_result.script["blocks"][si].get("image") or []
        img_paths: list[Path] = []
        for p in block_imgs:
            pp = Path(p)
            if not pp.is_absolute():
                pp = PROJECT_ROOT / pp
            if pp.exists():
                img_paths.append(pp)
        if not img_paths:
            raise RuntimeError(
                f"section {si+1} ('{section.get('title','')}') has no usable images. "
                f"Refine image_keywords and re-run."
            )
        mood = (section.get("mood") or "").strip().lower()
        if mood:
            moods.append(mood)
        chapters_for_editor.append({
            "title": section.get("title", f"Section {si+1}"),
            "start": start,
            "end": end,
            "images": img_paths,
        })

    chosen_mood = Counter(moods).most_common(1)[0][0] if moods else (default_mood or "")
    log.info("Dominant mood: %s", chosen_mood or "(none)")

    # === PHASE 6: COMPOSE ===
    log.info("=== EDITOR PHASE ===")
    output_path = PROJECT_ROOT / "output" / "videos" / f"{output_name}.mp4"
    result = compose_card(
        narration_wav=narration_wav,
        chapters=chapters_for_editor,
        words=global_words,
        output_path=output_path,
        mood=chosen_mood,
        emit=_emit,
    )

    # === PHASE 7: THUMBNAIL + METADATA ===
    log.info("=== THUMBNAIL + METADATA ===")
    try:
        from src.thumbnail import generate_thumbnail, write_youtube_metadata
        thumb = generate_thumbnail(video_path=output_path, title=topic, run_dir=run_dir)
        log.info("Thumbnail: %s", thumb)
        meta_out = write_youtube_metadata(saved_script_path)
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
