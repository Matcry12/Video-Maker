"""Podcast-style two-voice renderer using Chatterbox + Remotion 16:9 render.

Usage:
    .venv/bin/python scripts/render_podcast.py --script lab/podcast_script.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import torchaudio as ta
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tts_chatterbox import _ChatterboxBackend

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("podcast")

TURN_GAP_SEC = 0.2
REMOTION_DIR = ROOT / "remotion"
DEFAULT_SPEAKER_MAP = {"heart": "maya", "fenrir": "marcus"}


THUMB_W, THUMB_H = 1280, 720
_FONT_BOLD = ROOT / "assets" / "fonts" / "ChangaOne-Regular.ttf"


def _generate_podcast_thumbnail(video_path: Path, title: str) -> Path:
    out = video_path.parent / f"{video_path.stem}_thumb.jpg"

    # extract frame at 20% into video
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip() or "10")
    ts = duration * 0.20

    tmp = out.parent / "_thumb_frame.jpg"
    subprocess.run(
        ["ffmpeg", "-ss", str(ts), "-i", str(video_path),
         "-frames:v", "1", "-q:v", "2", str(tmp), "-y"],
        check=True, capture_output=True,
    )

    img = Image.open(tmp).convert("RGB")
    img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)

    # dark gradient at bottom third
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    grad_top = THUMB_H * 2 // 3
    for y in range(grad_top, THUMB_H):
        alpha = int(200 * (y - grad_top) / (THUMB_H - grad_top))
        draw_ov.line([(0, y), (THUMB_W, y)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # title text
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(str(_FONT_BOLD), size=80)
    except Exception:
        font = ImageFont.load_default()

    words = title.split()
    lines, cur = [], []
    for w in words:
        cur.append(w)
        bbox = draw.textbbox((0, 0), " ".join(cur), font=font)
        if bbox[2] - bbox[0] > THUMB_W - 80:
            if len(cur) > 1:
                lines.append(" ".join(cur[:-1]))
                cur = [w]
    lines.append(" ".join(cur))

    line_h = draw.textbbox((0, 0), "Ay", font=font)[3] + 8
    total_h = line_h * len(lines)
    y = THUMB_H - total_h - 40

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (THUMB_W - (bbox[2] - bbox[0])) // 2
        # outline
        for dx, dy in [(-3, -3), (3, -3), (-3, 3), (3, 3)]:
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += line_h

    img.save(str(out), "JPEG", quality=92)
    tmp.unlink(missing_ok=True)
    return out


def ensure_output_dirs() -> None:
    (ROOT / "output" / "videos").mkdir(parents=True, exist_ok=True)
    (ROOT / "output" / "runs").mkdir(parents=True, exist_ok=True)


def load_script(path: Path) -> tuple[str, list[tuple[str, str, float]], dict]:
    spec = importlib.util.spec_from_file_location("_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    title = getattr(mod, "TITLE", path.stem)
    script = getattr(mod, "SCRIPT")
    meta = {
        "speaker_map": getattr(mod, "SPEAKER_MAP", DEFAULT_SPEAKER_MAP),
        "section_id":  getattr(mod, "SECTION_ID", "section_00"),
        "title_label": getattr(mod, "TITLE_LABEL", title.replace("_", " ").title()),
    }
    return title, script, meta


def _write_remotion_section(title: str, meta: dict, timings: list[dict], wav_src: Path) -> list[dict]:
    section_id = meta["section_id"]
    speaker_map = meta["speaker_map"]
    title_label = meta["title_label"]

    turns = [
        {
            "speaker": speaker_map.get(t["speaker"], t["speaker"]),
            "line": t["line"],
            "start": round(t["start"], 3),
            "duration": round(t["duration"], 3),
        }
        for t in timings
    ]

    section = {"title": title_label, "audioFile": f"{section_id}.wav", "turns": turns}

    data_dir = REMOTION_DIR / "src" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    section_path = data_dir / f"{section_id}.json"
    section_path.write_text(json.dumps(section, indent=2) + "\n")
    log.info("remotion section → %s", section_path)

    public_dir = REMOTION_DIR / "public"
    public_dir.mkdir(parents=True, exist_ok=True)
    dst_wav = public_dir / f"{section_id}.wav"
    shutil.copy2(wav_src, dst_wav)
    log.info("remotion audio   → %s", dst_wav)

    return turns


def _silence(sr: int, sec: float) -> torch.Tensor:
    return torch.zeros(1, int(sr * sec))


def _apply_fade(wav: torch.Tensor, sr: int, fade_ms: float = 25.0) -> torch.Tensor:
    n = int(sr * fade_ms / 1000)
    n = min(n, wav.shape[-1] // 2)
    if n < 2:
        return wav
    ramp = torch.linspace(0.0, 1.0, n)
    wav = wav.clone()
    wav[..., :n] *= ramp
    wav[..., -n:] *= ramp.flip(0)
    return wav


def _to_mono_2d(w: torch.Tensor) -> torch.Tensor:
    if w.dim() == 1:
        return w.unsqueeze(0)
    if w.shape[0] > 1:
        return w.mean(dim=0, keepdim=True)
    return w


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _write_youtube_metadata(title: str, meta: dict, timings: list[dict], out_dir: Path) -> None:
    from src.llm_client import chat_completion
    title_label = meta["title_label"]
    script_lines = "\n".join(
        f"[{_fmt_ts(t['start'])}] {t['speaker'].upper()}: {t['line']}" for t in timings
    )
    prompt = f"""You are a YouTube growth expert. Given this podcast conversation video, write YouTube metadata.

Title: {title_label}
Script with timestamps:
{script_lines}

Respond in exactly this format (no extra text):
TITLE
<compelling YouTube title, under 70 characters, no clickbait but curiosity-driving>

DESCRIPTION
<3-4 sentence description that hooks viewers, explains what they'll learn, ends with a call to action>

CHAPTERS
0:00 Intro
<M:SS Chapter name for each sub-topic based on when it starts in the script above>

#hashtag1 #hashtag2 #hashtag3 #hashtag4 #hashtag5

TAGS
<15-20 comma-separated tags, mix of broad and specific, no #>"""

    response = chat_completion(
        system="You write YouTube metadata that drives clicks and watch time. Be concise and specific.",
        user=prompt,
        stage="script",
        temperature=0.7,
        max_tokens=700,
    )

    out_path = out_dir / "youtube_metadata.md"
    out_path.write_text(response.strip() + "\n", encoding="utf-8")
    log.info("YouTube metadata → %s", out_path)


def render(script_path: Path) -> Path:
    ensure_output_dirs()

    title, script, meta = load_script(script_path)
    out_dir = ROOT / "output" / "runs" / f"podcast_{title}"
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = _ChatterboxBackend.get()
    sr = backend.sr
    backend.cfg = 0.1

    segments: list[torch.Tensor] = []
    timings: list[dict] = []
    cursor = 0.0

    for i, (speaker, line, exag) in enumerate(script):
        out_wav = out_dir / f"turn_{i:02d}_{speaker}.wav"
        log.info("[%d/%d] %s (exag=%.2f): %r", i + 1, len(script), speaker.upper(), exag, line)

        backend.exag = exag
        t0 = time.time()
        duration, _ = backend.synthesize_to_wav(line, voice=speaker, output_path=out_wav)
        gen_sec = time.time() - t0

        wav, _ = ta.load(str(out_wav))
        wav = _apply_fade(_to_mono_2d(wav), sr, fade_ms=50.0)
        segments.append(wav)
        timings.append({
            "turn": i, "speaker": speaker, "exag": exag,
            "line": line, "start": cursor, "duration": duration, "gen_sec": gen_sec,
        })
        cursor += duration
        log.info("  audio=%.2fs  gen=%.2fs  rt=%.2fx  exag=%.2f",
                 duration, gen_sec, duration / gen_sec if gen_sec else 0, exag)

        if i < len(script) - 1:
            segments.append(_silence(sr, TURN_GAP_SEC))
            cursor += TURN_GAP_SEC

    full = torch.cat(segments, dim=-1)
    out_path = out_dir / f"podcast_{title}.wav"
    ta.save(str(out_path), full, sr)
    total = full.shape[-1] / sr

    log.info("")
    log.info("=== Podcast complete: %s ===", title)
    log.info("turns:  %d  |  total: %.2fs  |  file: %s (%d bytes)",
             len(script), total, out_path, out_path.stat().st_size)
    log.info("")
    for t in timings:
        log.info("  [%02d] %-7s exag=%.2f @ %.2fs  dur=%.2fs — %s",
                 t["turn"], t["speaker"].upper(), t["exag"],
                 t["start"], t["duration"], t["line"][:70])

    timings_path = out_dir / f"timings_{title}.json"
    with open(timings_path, "w") as f:
        json.dump(timings, f, indent=2)
    log.info("timings → %s", timings_path)

    turns = _write_remotion_section(title, meta, timings, out_path)

    assert out_path.exists() and total > 5.0

    section_id = meta["section_id"]
    output_mp4 = ROOT / "output" / "videos" / f"{title}.mp4"
    props = {
        "title": meta["title_label"],
        "audioFile": f"{section_id}.wav",
        "turns": turns,
    }
    log.info("Running Remotion render → %s", output_mp4)
    subprocess.run(
        [
            "npx", "remotion", "render",
            "src/index.ts", "ConversationVideo",
            str(output_mp4.resolve()),
            f"--props={json.dumps(props)}",
        ],
        cwd=str(REMOTION_DIR),
        check=True,
    )
    log.info("Video → %s", output_mp4)

    _write_youtube_metadata(title, meta, timings, out_dir)

    try:
        thumb = _generate_podcast_thumbnail(output_mp4, meta["title_label"])
        log.info("Thumbnail → %s", thumb)
    except Exception as exc:
        log.warning("Thumbnail generation failed (non-fatal): %s", exc)

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script",
        default=str(ROOT / "lab" / "podcast_script.py"),
        help="Path to script module (default: lab/podcast_script.py)",
    )
    args = parser.parse_args()
    render(Path(args.script))
    return 0


if __name__ == "__main__":
    sys.exit(main())
