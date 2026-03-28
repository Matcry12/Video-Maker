"""
Editor Module — FFmpeg video composition, subtitle generation, and rendering.

Supports per-block rendering with:
- Video or image backgrounds (Ken Burns effect for images)
- Overlay images (top-center popup with fade in/out)
- ASS subtitle burn-in
- NVENC hardware-accelerated encoding
- Clip concatenation and BGM mixing
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

import soundfile as sf

logger = logging.getLogger(__name__)

STYLES_ASS = Path(__file__).parent.parent / "styles.ass"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi"}
PUNCT_NO_SPACE_BEFORE = set(",.!?;:%)]}\"'")
PUNCT_NO_SPACE_AFTER = set("([{")
HARD_BREAK_PUNCT = (".", "!", "?", ";", ":", "…")
BAD_LINE_START_WORDS = {
    "a", "an", "and", "của", "for", "in", "là", "mà", "of", "or",
    "ở", "the", "thì", "to", "và", "với", "về",
}


def generate_ass(
    words: list,
    output_path: Path,
    style: str = "Default",
    highlight_style: str = "Highlight",
    subtitle_mode: str = "standard",
    highlight_keywords: Optional[list[str]] = None,
    max_chars_per_line: int = 28,
    max_lines_per_caption: int = 2,
    min_duration: float = 1.0,
    max_duration: float = 2.5,
    max_cps: float = 16.0,
    pause_break_sec: float = 0.35,
) -> Path:
    """
    Generate an .ass subtitle file from word timestamps.
    Uses phrase-based caption segmentation and pacing guardrails.
    """
    subtitle_mode = subtitle_mode.lower().strip()
    if subtitle_mode not in {"standard", "progressive"}:
        subtitle_mode = "standard"

    keyword_set = {_normalize_keyword(kw) for kw in (highlight_keywords or [])}
    keyword_set.discard("")

    header = STYLES_ASS.read_text()

    events = []
    captions = _build_captions(
        words=words,
        max_chars_per_line=max_chars_per_line,
        max_lines_per_caption=max_lines_per_caption,
        min_duration=min_duration,
        max_duration=max_duration,
        max_cps=max_cps,
        pause_break_sec=pause_break_sec,
    )

    for caption in captions:
        caption_events = _caption_to_events(
            caption=caption,
            style=style,
            highlight_style=highlight_style,
            subtitle_mode=subtitle_mode,
            keyword_set=keyword_set,
            max_chars_per_line=max_chars_per_line,
            max_lines_per_caption=max_lines_per_caption,
        )
        for event_data in caption_events:
            start_ts = _seconds_to_ass(event_data["start"])
            end_ts = _seconds_to_ass(event_data["end"])
            text = _escape_ass_text(event_data["text"])
            event = f"Dialogue: 0,{start_ts},{end_ts},{style},,0,0,0,,{text}"
            events.append(event)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(events))
        f.write("\n")

    logger.info("Subtitle file written: %s (%d lines)", output_path, len(events))
    return output_path


def _build_captions(
    words: list,
    max_chars_per_line: int,
    max_lines_per_caption: int,
    min_duration: float,
    max_duration: float,
    max_cps: float,
    pause_break_sec: float,
) -> list[dict]:
    """Build phrase-level subtitle captions with timing and readability guardrails."""
    normalized_words = _normalize_words(words)
    if not normalized_words:
        return []

    max_caption_chars = max_chars_per_line * max_lines_per_caption
    segments: list[list[dict]] = []
    current_segment: list[dict] = []

    for word in normalized_words:
        if _should_break_before(
            current_segment=current_segment,
            next_word=word,
            max_caption_chars=max_caption_chars,
            max_chars_per_line=max_chars_per_line,
            max_lines_per_caption=max_lines_per_caption,
            max_duration=max_duration,
            max_cps=max_cps,
            pause_break_sec=pause_break_sec,
        ):
            segments.append(current_segment)
            current_segment = []
        current_segment.append(word)

    if current_segment:
        segments.append(current_segment)

    captions = []
    for segment in segments:
        text = _join_tokens([w["word"] for w in segment])
        lines = _wrap_caption_lines(
            text,
            max_chars_per_line=max_chars_per_line,
            max_lines=max_lines_per_caption,
        )
        captions.append(
            {
                "start": segment[0]["start"],
                "end": segment[-1]["end"],
                "text": r"\N".join(lines),
                "words": segment,
            }
        )

    _smooth_caption_gaps(captions, min_gap=0.06, max_gap=0.24)
    _ensure_min_caption_duration(captions, min_duration=min_duration, min_gap=0.06)
    return captions


def _normalize_words(words: list) -> list[dict]:
    """Clean token text and timestamps while preserving original timing order."""
    normalized = []
    for item in words:
        token = re.sub(r"\s+", " ", str(item.get("word", "") or "")).strip()
        if not token:
            continue

        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
        except (TypeError, ValueError):
            continue

        start = max(0.0, start)
        end = max(start, end)
        normalized.append({"word": token, "start": start, "end": end})
    return normalized


def _normalize_keyword(keyword: str) -> str:
    return re.sub(r"[^\wÀ-ỹ]", "", keyword or "").lower()


def _caption_to_events(
    caption: dict,
    style: str,
    highlight_style: str,
    subtitle_mode: str,
    keyword_set: set[str],
    max_chars_per_line: int,
    max_lines_per_caption: int,
) -> list[dict]:
    if subtitle_mode == "progressive":
        return _caption_to_progressive_events(
            caption=caption,
            style=style,
            highlight_style=highlight_style,
            keyword_set=keyword_set,
            max_chars_per_line=max_chars_per_line,
            max_lines_per_caption=max_lines_per_caption,
        )

    text = _apply_keyword_highlight(caption["text"], keyword_set, style, highlight_style)
    return [{"start": caption["start"], "end": caption["end"], "text": text}]


def _caption_to_progressive_events(
    caption: dict,
    style: str,
    highlight_style: str,
    keyword_set: set[str],
    max_chars_per_line: int,
    max_lines_per_caption: int,
) -> list[dict]:
    words = caption.get("words") or []
    if len(words) <= 1:
        text = _apply_keyword_highlight(caption["text"], keyword_set, style, highlight_style)
        return [{"start": caption["start"], "end": caption["end"], "text": text}]

    events = []
    for idx in range(len(words)):
        start = words[idx]["start"]
        if idx < len(words) - 1:
            end = words[idx + 1]["start"]
        else:
            end = caption["end"]

        end = min(max(end, start + 0.08), caption["end"])
        if end <= start:
            continue

        visible_tokens = [w["word"] for w in words[: idx + 1]]
        plain_text = _join_tokens(visible_tokens)
        lines = _wrap_caption_lines(
            plain_text,
            max_chars_per_line=max_chars_per_line,
            max_lines=max_lines_per_caption,
        )
        progressive_text = r"\N".join(lines)
        progressive_text = _apply_keyword_highlight(
            progressive_text,
            keyword_set,
            style,
            highlight_style,
        )
        events.append({"start": start, "end": end, "text": progressive_text})

    if not events:
        text = _apply_keyword_highlight(caption["text"], keyword_set, style, highlight_style)
        return [{"start": caption["start"], "end": caption["end"], "text": text}]
    return events


def _apply_keyword_highlight(
    text: str,
    keyword_set: set[str],
    style: str,
    highlight_style: str,
) -> str:
    """Apply single-color keyword emphasis using ASS style resets."""
    if not keyword_set:
        return text

    chunks = re.split(r"(\s+)", text)
    highlighted = []
    for chunk in chunks:
        if not chunk or chunk.isspace():
            highlighted.append(chunk)
            continue
        normalized = _normalize_keyword(chunk)
        if normalized in keyword_set:
            highlighted.append(f"{{\\r{highlight_style}}}{chunk}{{\\r{style}}}")
        else:
            highlighted.append(chunk)
    return "".join(highlighted)


def _should_break_before(
    current_segment: list[dict],
    next_word: dict,
    max_caption_chars: int,
    max_chars_per_line: int,
    max_lines_per_caption: int,
    max_duration: float,
    max_cps: float,
    pause_break_sec: float,
) -> bool:
    if not current_segment:
        return False

    previous = current_segment[-1]
    pause = next_word["start"] - previous["end"]
    if pause >= pause_break_sec:
        return True

    if previous["word"].endswith(HARD_BREAK_PUNCT) and pause >= 0.08:
        return True

    candidate = current_segment + [next_word]
    candidate_text = _join_tokens([w["word"] for w in candidate])
    candidate_duration = max(next_word["end"] - candidate[0]["start"], 0.001)
    char_count = _text_len(candidate_text)

    if char_count > max_caption_chars:
        return True
    if candidate_duration > max_duration:
        return True
    if char_count / candidate_duration > max_cps:
        return True

    wrapped = _wrap_caption_lines(
        candidate_text,
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines_per_caption,
        hard_cap=False,
    )
    return len(wrapped) > max_lines_per_caption


def _join_tokens(tokens: list[str]) -> str:
    """Join tokens with punctuation-aware spacing."""
    if not tokens:
        return ""

    joined_parts: list[str] = []
    for token in tokens:
        if not joined_parts:
            joined_parts.append(token)
            continue

        prev_last = joined_parts[-1][-1]
        next_first = token[0]
        if next_first in PUNCT_NO_SPACE_BEFORE or prev_last in PUNCT_NO_SPACE_AFTER:
            joined_parts[-1] += token
        else:
            joined_parts.append(token)
    return " ".join(joined_parts)


def _text_len(text: str) -> int:
    """Count readable characters (ignores spaces)."""
    return len(text.replace(" ", ""))


def _wrap_caption_lines(
    text: str,
    max_chars_per_line: int,
    max_lines: int,
    hard_cap: bool = True,
) -> list[str]:
    """Wrap caption text into readable lines and keep line count bounded."""
    words = text.split()
    if not words:
        return []

    lines = []
    current = words[0]
    for word in words[1:]:
        proposal = f"{current} {word}"
        if len(proposal) <= max_chars_per_line:
            current = proposal
        else:
            lines.append(current)
            current = word
    lines.append(current)

    if hard_cap and len(lines) > max_lines:
        head = lines[: max_lines - 1]
        tail = " ".join(lines[max_lines - 1 :])
        lines = head + [tail]

    return _rebalance_lines(lines)


def _rebalance_lines(lines: list[str]) -> list[str]:
    """Balance 2-line captions and avoid bad short-word starts on line 2."""
    if len(lines) != 2:
        return lines

    first_words = lines[0].split()
    second_words = lines[1].split()
    if not first_words or not second_words:
        return [line for line in lines if line]

    while len(first_words) > 1:
        first_len = len(" ".join(first_words))
        second_len = len(" ".join(second_words))
        second_starter = re.sub(r"[^\wÀ-ỹ]", "", second_words[0]).lower()

        needs_balance = first_len - second_len > 8
        bad_starter = second_starter in BAD_LINE_START_WORDS
        one_word_tail = len(second_words) == 1 and len(first_words) > 2
        if not (needs_balance or bad_starter or one_word_tail):
            break

        second_words.insert(0, first_words.pop())

    return [" ".join(first_words), " ".join(second_words)]


def _smooth_caption_gaps(captions: list[dict], min_gap: float, max_gap: float):
    """Smooth gaps between neighboring caption events without shifting starts."""
    for idx in range(len(captions) - 1):
        current = captions[idx]
        nxt = captions[idx + 1]
        gap = nxt["start"] - current["end"]

        if gap > max_gap:
            current["end"] += min(gap - max_gap, 0.30)
            continue

        if gap < min_gap:
            target_end = max(current["start"] + 0.12, nxt["start"] - min_gap)
            current["end"] = min(current["end"], target_end)

        current["end"] = max(current["end"], current["start"] + 0.12)


def _ensure_min_caption_duration(captions: list[dict], min_duration: float, min_gap: float):
    """Extend too-short captions where possible to avoid flash-like timing."""
    for idx, caption in enumerate(captions):
        desired_end = caption["start"] + min_duration
        if caption["end"] >= desired_end:
            continue

        if idx == len(captions) - 1:
            caption["end"] = desired_end
            continue

        next_start = captions[idx + 1]["start"]
        max_allowed_end = max(caption["end"], next_start - min_gap)
        caption["end"] = min(desired_end, max_allowed_end)
        caption["end"] = max(caption["end"], caption["start"] + 0.12)


def _escape_ass_text(text: str) -> str:
    """
    Escape braces to avoid accidental ASS override tags in subtitle text.
    Keeps known style-reset tags intact, e.g. {\rHighlight}.
    """
    placeholders = {}

    def _protect_tag(match: re.Match) -> str:
        key = f"__ASS_TAG_{len(placeholders)}__"
        placeholders[key] = match.group(0)
        return key

    protected = re.sub(r"\{\\r[^}]+\}", _protect_tag, text)
    escaped = protected.replace("{", r"\{").replace("}", r"\}")

    for key, value in placeholders.items():
        escaped = escaped.replace(key, value)
    return escaped


def _seconds_to_ass(seconds: float) -> str:
    """Convert seconds to ASS timestamp format H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _run_ffmpeg(cmd: list, label: str = "FFmpeg"):
    """Run an FFmpeg command and raise on failure."""
    logger.debug("%s command: %s", label, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("%s stderr:\n%s", label, result.stderr)
        raise RuntimeError(f"{label} failed (exit {result.returncode}): {result.stderr[-500:]}")


def render_block_clip(
    background_path: Path,
    audio_path: Path,
    subtitle_path: Path,
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    overlay_image: Optional[Path] = None,
    background_offset_sec: float = 0.0,
    use_nvenc: bool = True,
) -> Path:
    """
    Render a single block clip.

    - Video background: loops to match audio duration
      and starts at background_offset_sec for timeline continuity
    - Image background: Ken Burns zoom effect
    - Optional overlay image: top-center with fade in/out
    - Burns ASS subtitles
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get audio duration for overlay fade timing
    audio_info = sf.info(str(audio_path))
    duration = audio_info.duration

    cmd = ["ffmpeg", "-y"]
    filters = []
    input_idx = 0

    # --- Input 0: Background ---
    if _is_image(background_path):
        # Image: use loop for Ken Burns
        cmd += ["-loop", "1", "-i", str(background_path)]
    else:
        # Video: loop indefinitely, seek to cumulative timeline offset
        seek_sec = max(0.0, float(background_offset_sec))
        if seek_sec > 0.001:
            cmd += ["-ss", f"{seek_sec:.3f}"]
        cmd += ["-stream_loop", "-1", "-i", str(background_path)]
    bg_idx = input_idx
    input_idx += 1

    # --- Input 1: Audio ---
    cmd += ["-i", str(audio_path)]
    audio_idx = input_idx
    input_idx += 1

    # --- Input 2: Overlay image (optional) ---
    ovr_idx = None
    if overlay_image and overlay_image.exists():
        cmd += ["-i", str(overlay_image)]
        ovr_idx = input_idx
        input_idx += 1

    # --- Background filter ---
    if _is_image(background_path):
        # Ken Burns: slow zoom in from center
        filters.append(
            f"[{bg_idx}:v]scale=8000:-1,"
            f"zoompan=z='min(zoom+0.0005,1.3)'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d=1:s={width}x{height}:fps={fps},"
            f"setsar=1[bg]"
        )
    else:
        # Video: scale and crop
        filters.append(
            f"[{bg_idx}:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,fps={fps}[bg]"
        )

    # --- Subtitle burn-in ---
    sub_path_escaped = str(subtitle_path).replace("\\", "/").replace(":", "\\:")
    filters.append(f"[bg]ass='{sub_path_escaped}'[subbed]")

    # --- Overlay image ---
    if ovr_idx is not None:
        # Scale overlay to ~30% width, preserve aspect ratio
        ovr_width = int(width * 0.3)
        fade_dur = 0.4
        fade_in_start = 0.3
        fade_out_start = max(duration - 0.7, fade_in_start + fade_dur)

        filters.append(
            f"[{ovr_idx}:v]scale={ovr_width}:-1,format=rgba,"
            f"fade=t=in:st={fade_in_start}:d={fade_dur}:alpha=1,"
            f"fade=t=out:st={fade_out_start:.2f}:d={fade_dur}:alpha=1[ovr]"
        )
        # Position: top-center with padding
        filters.append(
            f"[subbed][ovr]overlay=x=(W-w)/2:y=50:"
            f"enable='between(t,{fade_in_start},{fade_out_start + fade_dur:.2f})'[vout]"
        )
    else:
        filters.append("[subbed]copy[vout]")

    # --- Build command ---
    filter_complex = ";\n".join(filters)
    cmd += ["-filter_complex", filter_complex]
    cmd += ["-map", "[vout]", "-map", f"{audio_idx}:a"]
    cmd += ["-shortest"]

    # Encoding
    if use_nvenc:
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]
    cmd += ["-c:a", "aac", "-b:a", "192k"]

    cmd += [str(output_path)]

    logger.info("Rendering block clip: %s", output_path.name)
    _run_ffmpeg(cmd, "render_block_clip")
    logger.info("Block clip done: %s", output_path.name)
    return output_path


def concat_clips(clip_paths: list[Path], output_path: Path) -> Path:
    """Concatenate multiple clips into one video using FFmpeg concat demuxer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write concat list file
    list_file = output_path.parent / f"{output_path.stem}_concat.txt"
    with open(list_file, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{clip.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]

    logger.info("Concatenating %d clips...", len(clip_paths))
    _run_ffmpeg(cmd, "concat_clips")

    # Cleanup list file
    list_file.unlink(missing_ok=True)
    logger.info("Concat done: %s", output_path)
    return output_path


def mix_bgm(
    video_path: Path,
    bgm_path: Path,
    output_path: Path,
    bgm_volume: float = 0.15,
) -> Path:
    """Mix background music into a video. Video stream is copied, only audio re-encoded."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-stream_loop", "-1", "-i", str(bgm_path),
        "-filter_complex",
        f"[1:a]volume={bgm_volume}[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ]

    logger.info("Mixing BGM...")
    _run_ffmpeg(cmd, "mix_bgm")
    logger.info("BGM mixed: %s", output_path)
    return output_path
