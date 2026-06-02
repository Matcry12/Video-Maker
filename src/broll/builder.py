"""B-roll Shorts pipeline orchestrator.

Entry point for the Psychology Facts channel engine. Converts a topic string
into a finished mp4 with narration, stock footage, animated karaoke captions,
BGM, and YouTube metadata.

Usage (programmatic):
    from src.broll import build_broll
    result = build_broll("Why dopamine lies to you")

Usage (CLI):
    ./.venv/bin/python -m src.broll.builder --topic "Why dopamine lies to you"
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root is on sys.path when run as __main__
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env (PEXELS_API_KEY, GROQ_API_KEY, …)
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except Exception:
    pass

from src.broll import compose, keywords, remotion_render
from src.broll import content_index
from src.editor import _lab_mix_bgm, _lab_pick_bgm
from src.tts import TTSEngine, resolve_tts_voice

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("broll.builder")

# Channel default BGM (Psychology Facts niche). Falls back to mood-pick if missing.
DEFAULT_BGM = Path("/home/matcry/AFTER HOURS INSTRUMENTAL (Slowed Reverb) [HnFGQ_kOGcc].mp3")

# Output directory for final videos.
_OUT_DIR = _PROJECT_ROOT / "output" / "broll"

# B-roll visual layout (square bare → Remotion places it on paper background).
_LAYOUT = "square_bare"
_BG_COLOR = "0x1565C0"


class BrollDuplicateError(Exception):
    """Raised when the topic has already been produced (found in content_index)."""
    def __init__(self, topic: str) -> None:
        super().__init__(f"Topic already produced: {topic!r}")
        self.topic = topic


@dataclass
class BrollResult:
    final_video: Path
    work_dir: Path
    topic: str
    hook: str
    beats: list[str]
    queries: list[list[str]]
    youtube: dict
    duration: float


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "broll"


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _concat_wavs(wavs: list[Path], out: Path) -> None:
    """Concatenate WAVs by raw PCM frames (avoids inter-chunk silence)."""
    params = None
    frames = []
    for p in wavs:
        with wave.open(str(p), "rb") as w:
            if params is None:
                params = w.getparams()
            frames.append(w.readframes(w.getnframes()))
    with wave.open(str(out), "wb") as w:
        w.setparams(params)
        for fr in frames:
            w.writeframes(fr)


def build_broll(
    topic: str,
    *,
    target_words: int = 130,
    bgm_vol: float = 0.16,
    skip_dupe_check: bool = False,
    emit=None,
) -> BrollResult:
    """Full pipeline: dedupe-check -> script -> hook -> queries -> TTS -> footage
    -> Remotion render -> mux narration -> BGM -> metadata -> append to index.

    Returns BrollResult; raises BrollDuplicateError if topic already produced.

    Args:
        topic: The video subject (e.g. "Why dopamine lies to you").
        target_words: Approximate narration word count.
        bgm_vol: BGM mix volume (0.0–1.0). Default 0.16.
        skip_dupe_check: If True, skip the duplicate-topic guard.
        emit: Optional progress callback emit(stage: str, msg: str).
    """
    def _emit(stage: str, msg: str) -> None:
        logger.info("[%s] %s", stage, msg)
        if emit is not None:
            emit(stage, msg)

    # --- Duplicate check ---
    if not skip_dupe_check and content_index.is_duplicate(topic):
        raise BrollDuplicateError(topic)

    # --- Work directory ---
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{_slug(topic)}_{int(time.time())}"
    work_dir = _OUT_DIR / name
    work_dir.mkdir(parents=True, exist_ok=True)
    _emit("init", f"Topic: {topic!r}  work_dir: {work_dir}")

    # --- 1. Script -> beats ---
    _emit("script", "Writing script...")
    beats = keywords.write_script(topic, target_words=target_words)
    _emit("script", f"{len(beats)} beats generated")

    hook_text = keywords.clickbait_hook(topic)
    _emit("script", f"Hook: {hook_text!r}")

    # --- 2. Per-beat visual queries ---
    _emit("queries", "Generating visual queries...")
    queries = keywords.visual_queries(beats, topic)
    for i, qs in enumerate(queries):
        logger.info("  beat %d -> %s", i, qs)

    # --- 3. TTS per beat ---
    _emit("tts", "Synthesising narration...")
    voice = resolve_tts_voice(language="en")
    tts = TTSEngine()
    beat_audio: list[Path] = []
    beat_durs: list[float] = []
    global_words: list[dict] = []
    cursor = 0.0
    try:
        for i, beat in enumerate(beats):
            wav = work_dir / f"beat{i}.wav"
            synth = tts.synthesize(
                text=beat, output_path=wav, voice=voice,
                rate="+18%", alignment_mode="corrected",
            )
            dur = float(synth.get("duration") or 0.0)
            if dur <= 0:
                dur = _wav_duration(wav)
            for w in synth.get("words", []):
                tok = str(w.get("word", "")).strip()
                if not tok:
                    continue
                global_words.append({
                    "word": tok,
                    "start": cursor + float(w.get("start", 0.0)),
                    "end": cursor + float(w.get("end", w.get("start", 0.0))),
                })
            beat_audio.append(wav)
            beat_durs.append(dur)
            cursor += dur
            logger.info("  beat %d: %.2fs", i, dur)
    finally:
        tts.close()

    total_dur = sum(beat_durs)
    _emit("tts", f"Total narration: {total_dur:.2f}s")

    narration = work_dir / "narration.wav"
    _concat_wavs(beat_audio, narration)

    # --- 4. Fetch + build b-roll segments ---
    _emit("footage", "Fetching + building b-roll segments...")
    beat_videos: list[Path] = []
    used_ids: set = set()
    for i, dur in enumerate(beat_durs):
        seg = compose.build_beat_segment(
            i, queries[i], dur, work_dir, layout=_LAYOUT, bg_color=_BG_COLOR,
            used_ids=used_ids,
        )
        if seg is None:
            logger.warning("  beat %d: no footage -> reusing previous", i)
            if beat_videos:
                seg = work_dir / f"beat{i}.mp4"
                compose._exact_duration(beat_videos[-1], seg, dur)
            else:
                raise RuntimeError("beat 0 has no footage; cannot continue")
        beat_videos.append(seg)

    background = compose.concat_background(beat_videos, work_dir)
    _emit("footage", f"Background built: {background.name}")

    # --- 5. Remotion render ---
    _emit("remotion", "Remotion render (paper + square + karaoke)...")
    props = remotion_render.prepare_assets(
        background, global_words, fps=30, total_dur=total_dur, hook_text=hook_text,
    )
    animated = work_dir / "animated.mp4"
    remotion_render.render(props, animated)

    # Mux narration audio onto the silent Remotion video
    with_voice = work_dir / "with_voice.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(animated),
            "-i", str(narration),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            str(with_voice),
        ],
        check=True,
    )
    _emit("remotion", f"Audio muxed: {with_voice.name}")

    # --- 6. BGM mix ---
    _emit("bgm", "Mixing BGM...")
    final = _OUT_DIR / f"{name}.mp4"
    bgm_dir = _PROJECT_ROOT / "assets" / "audio" / "bgm"
    bgm = DEFAULT_BGM if DEFAULT_BGM.exists() else _lab_pick_bgm(bgm_dir, category="calm")
    if bgm is not None:
        _lab_mix_bgm(with_voice, bgm, final, bgm_vol=bgm_vol)
        _emit("bgm", f"BGM: {bgm.name}")
    else:
        import shutil
        shutil.copy2(with_voice, final)
        _emit("bgm", "No BGM found, skipping")

    # --- 7. Metadata + index ---
    _emit("metadata", "Generating YouTube metadata...")
    meta = keywords.generate_metadata(topic, beats, hook_text)

    # Write human-readable metadata file
    meta_txt = work_dir / "youtube_metadata.txt"
    meta_txt.write_text(
        f"TITLE\n{meta['title']}\n\n"
        f"DESCRIPTION\n{meta['description']}\n\n"
        f"TAGS\n{', '.join(meta['tags'])}\n",
        encoding="utf-8",
    )
    _emit("metadata", f"Metadata written: {meta_txt.name}")

    content_index.append_entry(
        topic=topic,
        niche="Psychology Facts",
        hook=hook_text,
        output_file=str(final),
        bgm=bgm.name if bgm is not None else "",
        youtube=meta,
        status="draft",
    )
    _emit("metadata", "Content index updated")

    _emit("done", f"FINAL VIDEO: {final}")

    return BrollResult(
        final_video=final,
        work_dir=work_dir,
        topic=topic,
        hook=hook_text,
        beats=beats,
        queries=queries,
        youtube=meta,
        duration=total_dur,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="B-roll Shorts pipeline")
    ap.add_argument("--topic", required=True, help="Video topic")
    ap.add_argument("--words", type=int, default=130, help="Target script word count")
    ap.add_argument("--bgm-vol", type=float, default=0.16, help="BGM volume (0–1)")
    ap.add_argument(
        "--force", action="store_true",
        help="Skip duplicate-topic check (skip_dupe_check=True)",
    )
    args = ap.parse_args()

    try:
        result = build_broll(
            args.topic,
            target_words=args.words,
            bgm_vol=args.bgm_vol,
            skip_dupe_check=args.force,
        )
    except BrollDuplicateError as exc:
        print(f"\nALREADY PRODUCED: {exc.topic!r} is already in the content index.")
        print("Use --force to produce it again anyway.")
        return 3

    print(f"\nFINAL VIDEO: {result.final_video}")
    print(f"\nYouTube metadata:")
    print(f"  Title:       {result.youtube.get('title', '')}")
    print(f"  Tags:        {', '.join(result.youtube.get('tags', []))}")
    print(f"  Duration:    {result.duration:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
