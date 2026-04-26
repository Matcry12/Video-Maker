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

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

STYLES_ASS = Path(__file__).parent.parent / "styles.ass"
POP_SFX = Path(__file__).parent.parent / "assets" / "audio" / "sound_effects" / "pop.mp3"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi"}
PUNCT_NO_SPACE_BEFORE = set(",.!?;:%)]}\"'")
PUNCT_NO_SPACE_AFTER = set("([{")
HARD_BREAK_PUNCT = (".", "!", "?", ";", ":", "…")
BAD_LINE_START_WORDS = {
    "a", "an", "and", "của", "for", "in", "là", "mà", "of", "or",
    "ở", "the", "thì", "to", "và", "với", "về",
}
TOKEN_VARIANT_TABLE = str.maketrans(
    {
        "“": "\"",
        "”": "\"",
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "…": "...",
    }
)
from .agent_config import load_agent_settings as _load_agent_settings
_editor_cfg = _load_agent_settings().get("editor", {})

MIN_EVENT_DURATION_SEC = float(_editor_cfg.get("min_event_duration_sec", 0.35))
_MAX_OVERLAYS = int(_editor_cfg.get("max_overlays", 12))
_FADE_DURATION_SEC = float(_editor_cfg.get("fade_duration_sec", 0.15))
_SCALE_FACTOR = float(_editor_cfg.get("scale_factor", 1.15))

# Pan direction presets for background image mode
# (start_x_frac, start_y_frac, end_x_frac, end_y_frac) of max pan distance
_BG_PAN_DIRS = [
    (0.0, 0.0, 1.0, 1.0),     # top-left → bottom-right
    (1.0, 0.0, 0.0, 1.0),     # top-right → bottom-left
    (0.0, 1.0, 1.0, 0.0),     # bottom-left → top-right
    (1.0, 1.0, 0.0, 0.0),     # bottom-right → top-left
    (0.5, 0.0, 0.5, 1.0),     # top-center → bottom-center
    (0.0, 0.5, 1.0, 0.5),     # left-center → right-center
]


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
    audio_duration: Optional[float] = None,
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

    # Clamp last caption end time to audio duration if provided
    if audio_duration is not None and audio_duration > 0 and captions:
        last = captions[-1]
        if last["end"] > audio_duration:
            last["end"] = audio_duration
        # Also ensure no caption exceeds audio duration
        for cap in captions:
            if cap["end"] > audio_duration:
                cap["end"] = audio_duration
            if cap["start"] >= audio_duration:
                cap["start"] = max(0.0, audio_duration - 0.1)
        # Extend last caption to fill audio — prevents blank video at the end
        # (TTS tail margin leaves the last word ending slightly before audio_duration)
        if captions[-1]["end"] < audio_duration:
            captions[-1]["end"] = audio_duration

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
            min_duration=min_duration,
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
        token = token.translate(TOKEN_VARIANT_TABLE)
        if not token:
            continue

        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
        except (TypeError, ValueError):
            continue

        start = max(0.0, start)
        end = max(start, end)

        token_parts = _split_token_with_timing(token, start, end)
        for part in token_parts:
            if normalized:
                prev = normalized[-1]
                same_word = prev["word"] == part["word"]
                same_start = abs(prev["start"] - part["start"]) < 0.001
                same_end = abs(prev["end"] - part["end"]) < 0.001
                if same_word and same_start and same_end:
                    continue
            normalized.append(part)
    return normalized


def _split_token_with_timing(token: str, start: float, end: float) -> list[dict]:
    """
    Edge may occasionally emit multi-word chunks; split them so caption
    segmentation never drops embedded words.
    """
    parts = [p for p in token.split(" ") if p]
    if len(parts) <= 1:
        return [{"word": token, "start": start, "end": end}]

    total_span = max(end - start, 0.001)
    weights = [max(len(re.sub(r"\W+", "", part)), 1) for part in parts]
    total_weight = max(sum(weights), 1)

    cursor = start
    result = []
    for idx, (part, weight) in enumerate(zip(parts, weights)):
        if idx == len(parts) - 1:
            part_end = end
        else:
            part_end = cursor + total_span * (weight / total_weight)
        part_end = max(part_end, cursor)
        result.append({"word": part, "start": cursor, "end": part_end})
        cursor = part_end
    return result


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

        end = min(max(end, start + MIN_EVENT_DURATION_SEC), caption["end"])
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
    min_duration: float,
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
    # Edge word boundaries can be very short at token-level. Use a duration floor
    # to avoid over-splitting into 1-word captions in standard mode.
    effective_duration = max(candidate_duration, max(min_duration, MIN_EVENT_DURATION_SEC))
    if len(candidate) >= 3 and (char_count / effective_duration > max_cps):
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
            target_end = max(current["start"] + MIN_EVENT_DURATION_SEC, nxt["start"] - min_gap)
            current["end"] = min(current["end"], target_end)

        current["end"] = max(current["end"], current["start"] + MIN_EVENT_DURATION_SEC)


def _ensure_min_caption_duration(captions: list[dict], min_duration: float, min_gap: float):
    """Extend too-short captions where possible to avoid flash-like timing."""
    min_duration = max(min_duration, MIN_EVENT_DURATION_SEC)
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
        caption["end"] = max(caption["end"], caption["start"] + MIN_EVENT_DURATION_SEC)


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
    overlay_images: Optional[list[dict]] = None,
    image_mode: str = "popup",
) -> Path:
    """
    Render a single block clip.

    - Video background: loops to match audio duration
    - Image background: Ken Burns zoom effect
    - Multi-image overlays with popup or background display modes
    - Burns ASS subtitles
    - Pop sound pre-mixed into audio for popup mode

    overlay_images: list of {"path": Path, "start_sec": float, "end_sec": float}
                    Times are relative to block start (0-based).
    image_mode: "popup" (floating overlay + pop sound) or "background" (full-screen pan)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_info = sf.info(str(audio_path))
    duration = audio_info.duration

    # Collect valid overlays (prefer new multi-image param, fall back to legacy single)
    valid_overlays = []
    if overlay_images:
        valid_overlays = [
            ovr for ovr in overlay_images
            if ovr.get("path") and Path(ovr["path"]).exists()
        ][:_MAX_OVERLAYS]
    elif overlay_image and overlay_image.exists():
        valid_overlays = [{"path": overlay_image, "start_sec": 0.05, "end_sec": duration}]

    # Pre-mix pop sounds into block audio (popup mode only)
    actual_audio = audio_path
    if image_mode == "popup" and POP_SFX.exists() and valid_overlays:
        try:
            pop_times = [float(ovr.get("start_sec", 0)) for ovr in valid_overlays]
            actual_audio = _premix_pop_sounds(audio_path, pop_times)
        except Exception as exc:
            logger.warning("Pop sound pre-mix failed: %s. Using original audio.", exc)

    cmd = ["ffmpeg", "-y"]
    filters = []
    input_idx = 0

    # --- Input 0: Background ---
    # Background mode with images: use solid black — video is NEVER visible
    use_black_bg = (image_mode == "background" and valid_overlays)

    if use_black_bg:
        cmd += ["-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration}"]
    elif _is_image(background_path):
        cmd += ["-loop", "1", "-i", str(background_path)]
    else:
        seek_sec = max(0.0, float(background_offset_sec))
        if seek_sec > 0.001:
            cmd += ["-ss", f"{seek_sec:.3f}"]
        cmd += ["-stream_loop", "-1", "-i", str(background_path)]
    input_idx += 1

    # --- Input 1: Audio ---
    cmd += ["-i", str(actual_audio)]
    audio_idx = input_idx
    input_idx += 1

    # --- Background filter ---
    if use_black_bg:
        filters.append(f"[0:v]setsar=1[bg]")
    elif _is_image(background_path):
        filters.append(
            f"[0:v]scale=8000:-1,"
            f"zoompan=z='min(zoom+0.0005,1.3)'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d=1:s={width}x{height}:fps={fps},"
            f"setsar=1[bg]"
        )
    else:
        filters.append(
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,fps={fps}[bg]"
        )

    # --- Subtitle burn-in & Overlay images ---
    # Background mode: overlay images FIRST (full-screen), then burn subtitles ON TOP
    # Popup mode: burn subtitles first, then overlay images on top
    sub_path_escaped = str(subtitle_path).replace("\\", "/").replace(":", "\\:")

    if valid_overlays and image_mode == "background":
        # Background mode: images on [bg] first, subtitles last
        current_label = "bg"

        # Full screen, no gaps, hard cuts — Ken Burns pan effect
        bg_scale_w = int(width * 1.15)
        bg_scale_h = int(height * 1.15)
        x_pan_max = bg_scale_w - width
        y_pan_max = bg_scale_h - height

        for oi, ovr in enumerate(valid_overlays):
            start_sec = float(ovr.get("start_sec", 0))
            end_sec = float(ovr.get("end_sec", start_sec + 4.0))
            dur = max(end_sec - start_sec, 0.1)

            cmd += ["-loop", "1", "-i", str(Path(ovr["path"]))]
            ovr_input = input_idx
            input_idx += 1

            sx, sy, ex, ey = _BG_PAN_DIRS[oi % len(_BG_PAN_DIRS)]
            sx_px = int(sx * x_pan_max)
            sy_px = int(sy * y_pan_max)
            ex_px = int(ex * x_pan_max)
            ey_px = int(ey * y_pan_max)

            ovr_label = f"ovr{oi}"
            # Last overlay goes to "img_composed", then subtitles go on top
            is_last_ovr = (oi == len(valid_overlays) - 1)
            out_label = "img_composed" if is_last_ovr else f"v{oi}"

            # No alpha fade — hard cut between images for seamless transitions
            filters.append(
                f"[{ovr_input}:v]scale={bg_scale_w}:{bg_scale_h}"
                f":force_original_aspect_ratio=increase,"
                f"crop={bg_scale_w}:{bg_scale_h},format=rgba[{ovr_label}]"
            )

            prog = f"min(max((t-{start_sec:.2f})/{dur:.2f},0),1)"
            x_expr = f"-({sx_px}+({ex_px}-{sx_px})*{prog})"
            y_expr = f"-({sy_px}+({ey_px}-{sy_px})*{prog})"

            # enable covers exact time range — no gaps between consecutive images
            filters.append(
                f"[{current_label}][{ovr_label}]overlay="
                f"x='{x_expr}':y='{y_expr}':"
                f"enable='between(t,{start_sec:.2f},{end_sec:.2f})'[{out_label}]"
            )
            current_label = out_label

        # Burn subtitles ON TOP of composed images so text is always visible
        filters.append(f"[img_composed]ass='{sub_path_escaped}'[vout]")

    elif valid_overlays:
        # Popup mode: subtitles first, then floating image overlays on top
        filters.append(f"[bg]ass='{sub_path_escaped}'[subbed]")
        current_label = "subbed"

        ovr_width = int(width * 0.68)
        for oi, ovr in enumerate(valid_overlays):
            start_sec = float(ovr.get("start_sec", 0))
            end_sec = float(ovr.get("end_sec", start_sec + 3.0))

            cmd += ["-loop", "1", "-i", str(Path(ovr["path"]))]
            ovr_input = input_idx
            input_idx += 1

            fade_dur = _FADE_DURATION_SEC
            fade_in_start = start_sec + 0.05
            fade_out_start = max(end_sec - 0.25, fade_in_start + fade_dur)
            overlay_end = fade_out_start + fade_dur

            ovr_label = f"ovr{oi}"
            out_label = f"v{oi}" if oi < len(valid_overlays) - 1 else "vout"

            filters.append(
                f"[{ovr_input}:v]scale={ovr_width}:-1,format=rgba,"
                f"fade=t=in:st={fade_in_start:.2f}:d={fade_dur}:alpha=1,"
                f"fade=t=out:st={fade_out_start:.2f}:d={fade_dur}:alpha=1[{ovr_label}]"
            )
            float_y = f"(H-h)/2-40+6*sin((t-{start_sec:.2f})*1.5)"
            filters.append(
                f"[{current_label}][{ovr_label}]overlay="
                f"x=(W-w)/2:y='{float_y}':"
                f"enable='between(t,{fade_in_start:.2f},{overlay_end:.2f})'[{out_label}]"
            )
            current_label = out_label
    else:
        # No overlays: just burn subtitles
        filters.append(f"[bg]ass='{sub_path_escaped}'[vout]")

    # --- Build command ---
    filter_complex = ";\n".join(filters)
    cmd += ["-filter_complex", filter_complex]
    cmd += ["-map", "[vout]", "-map", f"{audio_idx}:a"]
    cmd += ["-shortest"]

    if use_nvenc:
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]
    cmd += ["-c:a", "aac", "-b:a", "192k"]

    cmd += [str(output_path)]

    logger.info("Rendering block clip (%s mode, %d overlays): %s",
                image_mode, len(valid_overlays), output_path.name)
    _run_ffmpeg(cmd, "render_block_clip")

    # Clean up pre-mixed audio
    if actual_audio != audio_path and actual_audio.exists():
        actual_audio.unlink(missing_ok=True)

    logger.info("Block clip done: %s", output_path.name)
    return output_path


def render_block_clip_fast(
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
    overlay_images: Optional[list[dict]] = None,
    image_mode: str = "popup",
) -> Path:
    """Fast renderer: pre-composes frames with PIL, pipes rawvideo to FFmpeg.

    For background mode with overlay images, this is 5-10x faster than
    render_block_clip() because it avoids the N-layer FFmpeg overlay filter chain.
    Instead, PIL composes one frame at a time and pipes raw RGB to FFmpeg stdin.

    Falls through to render_block_clip() for popup mode or no overlays.
    """
    from PIL import Image as PILImage

    # Only use fast path for background mode with overlays
    # No _MAX_OVERLAYS cap here: PIL pre-composition iterates frames, it has no
    # N-layer filter-chain memory pressure that the slow path has.
    valid_overlays = []
    if overlay_images:
        valid_overlays = [
            ovr for ovr in overlay_images
            if ovr.get("path") and Path(ovr["path"]).exists()
        ]
    elif overlay_image and overlay_image.exists():
        valid_overlays = [{"path": overlay_image, "start_sec": 0.05, "end_sec": 99999.0}]

    if image_mode != "background" or not valid_overlays:
        return render_block_clip(
            background_path=background_path, audio_path=audio_path,
            subtitle_path=subtitle_path, output_path=output_path,
            width=width, height=height, fps=fps, overlay_image=overlay_image,
            background_offset_sec=background_offset_sec, use_nvenc=use_nvenc,
            overlay_images=overlay_images, image_mode=image_mode,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_info = sf.info(str(audio_path))
    duration = audio_info.duration
    total_frames = int(duration * fps)

    # Pre-mix pop sounds (not used in background mode, but keep for consistency)
    actual_audio = audio_path

    # --- Pre-load all overlay images as PIL ---
    # Scale to 1.15x output for Ken Burns pan headroom
    scale_factor = _SCALE_FACTOR
    scaled_w = int(width * scale_factor)
    scaled_h = int(height * scale_factor)

    loaded_images: list[dict] = []
    for oi, ovr in enumerate(valid_overlays):
        img_path = Path(ovr["path"])
        start_sec = float(ovr.get("start_sec", 0))
        end_sec = float(ovr.get("end_sec", start_sec + 4.0))

        try:
            img = PILImage.open(img_path).convert("RGB")
            # Scale to cover the scaled dimensions
            img_ratio = max(scaled_w / img.width, scaled_h / img.height)
            new_w = int(img.width * img_ratio)
            new_h = int(img.height * img_ratio)
            img = img.resize((new_w, new_h), PILImage.LANCZOS)
            # Center crop to exact scaled dimensions
            left = (new_w - scaled_w) // 2
            top = (new_h - scaled_h) // 2
            img = img.crop((left, top, left + scaled_w, top + scaled_h))
        except Exception as exc:
            logger.warning("Failed to load overlay image %s: %s", img_path, exc)
            continue

        # Pan direction
        sx, sy, ex, ey = _BG_PAN_DIRS[oi % len(_BG_PAN_DIRS)]
        x_pan_max = scaled_w - width
        y_pan_max = scaled_h - height

        loaded_images.append({
            "img": img,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "sx_px": int(sx * x_pan_max),
            "sy_px": int(sy * y_pan_max),
            "ex_px": int(ex * x_pan_max),
            "ey_px": int(ey * y_pan_max),
        })

    if not loaded_images:
        # All images failed to load — fall back
        return render_block_clip(
            background_path=background_path, audio_path=audio_path,
            subtitle_path=subtitle_path, output_path=output_path,
            width=width, height=height, fps=fps, overlay_image=overlay_image,
            background_offset_sec=background_offset_sec, use_nvenc=use_nvenc,
            overlay_images=overlay_images, image_mode=image_mode,
        )

    # Convert loaded PIL images to numpy arrays for fast cropping
    for item in loaded_images:
        item["np_img"] = np.array(item["img"])
        del item["img"]  # free PIL object

    # --- Build FFmpeg command: rawvideo stdin + audio + ASS subtitles ---
    sub_path_escaped = str(subtitle_path).replace("\\", "/").replace(":", "\\:")

    cmd = [
        "ffmpeg", "-y",
        # Input 0: raw RGB frames from stdin
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0",
        # Input 1: audio
        "-i", str(actual_audio),
        # Filter: burn subtitles on the video stream
        "-filter_complex", f"[0:v]ass='{sub_path_escaped}'[vout]",
        "-map", "[vout]", "-map", "1:a",
        "-shortest",
    ]

    if use_nvenc:
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]
    cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += [str(output_path)]

    logger.info(
        "Fast rendering: %d frames, %d overlays, %.1fs, %dx%d → %s",
        total_frames, len(loaded_images), duration, width, height, output_path.name,
    )

    # --- Pipe frames to FFmpeg ---
    import time as _time
    t_start = _time.monotonic()

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    # Black frame for gaps where no image is active
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    try:
        for frame_idx in range(total_frames):
            t = frame_idx / fps

            # Find which image is active at time t
            active_img = None
            for item in loaded_images:
                if item["start_sec"] <= t < item["end_sec"]:
                    active_img = item
                    break

            if active_img is None:
                # No image at this time — black frame
                proc.stdin.write(black_frame.tobytes())
                continue

            # Calculate Ken Burns pan position
            dur = active_img["end_sec"] - active_img["start_sec"]
            progress = min(max((t - active_img["start_sec"]) / dur, 0.0), 1.0)

            crop_x = int(active_img["sx_px"] + (active_img["ex_px"] - active_img["sx_px"]) * progress)
            crop_y = int(active_img["sy_px"] + (active_img["ey_px"] - active_img["sy_px"]) * progress)

            # Crop the output-sized region from the scaled image
            frame = active_img["np_img"][crop_y:crop_y + height, crop_x:crop_x + width]

            # Safety: if crop is out of bounds, pad with black
            if frame.shape[0] != height or frame.shape[1] != width:
                padded = black_frame.copy()
                h_actual = min(frame.shape[0], height)
                w_actual = min(frame.shape[1], width)
                padded[:h_actual, :w_actual] = frame[:h_actual, :w_actual]
                frame = padded

            proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else b""
            logger.error("render_block_clip_fast stderr:\n%s", stderr.decode()[-500:])
            raise RuntimeError(
                f"render_block_clip_fast failed (exit {proc.returncode}): "
                f"{stderr.decode()[-500:]}"
            )
    except BrokenPipeError:
        proc.kill()
        proc.wait()
        stderr = proc.stderr.read() if proc.stderr else b""
        raise RuntimeError(f"render_block_clip_fast pipe broken: {stderr.decode()[-500:]}")

    elapsed = _time.monotonic() - t_start
    logger.info("Fast render done in %.1fs: %s", elapsed, output_path.name)

    # Free numpy arrays
    for item in loaded_images:
        del item["np_img"]

    return output_path


def concat_audio(audio_paths: list[Path], output_path: Path) -> Path:
    """Concatenate multiple audio files into one seamless track."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    list_file = output_path.parent / f"{output_path.stem}_audiolist.txt"
    with open(list_file, "w") as f:
        for audio in audio_paths:
            f.write(f"file '{audio.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]

    logger.info("Concatenating %d audio files...", len(audio_paths))
    _run_ffmpeg(cmd, "concat_audio")
    list_file.unlink(missing_ok=True)
    logger.info("Audio concat done: %s", output_path)
    return output_path


def merge_ass_subtitles(
    subtitle_paths: list[Path],
    durations: list[float],
    output_path: Path,
) -> Path:
    """Merge multiple ASS subtitle files into one, offsetting timestamps."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read header from first file
    header_lines: list[str] = []
    first_events_start = -1
    first_content = subtitle_paths[0].read_text(encoding="utf-8")
    for i, line in enumerate(first_content.splitlines()):
        if line.strip().startswith("Dialogue:"):
            first_events_start = i
            break
        header_lines.append(line)

    # Collect all events with time offsets, clamping to block boundaries
    all_events: list[str] = []
    cumulative_offset = 0.0

    for idx, (sub_path, dur) in enumerate(zip(subtitle_paths, durations)):
        block_end = cumulative_offset + dur
        content = sub_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped.startswith("Dialogue:"):
                continue
            if cumulative_offset > 0.001:
                shifted = _shift_dialogue_line(stripped, cumulative_offset)
            else:
                shifted = stripped
            # Clamp event to block boundary to prevent overlap with next block
            shifted = _clamp_dialogue_to_boundary(shifted, block_end)
            if shifted:
                all_events.append(shifted)
        cumulative_offset += dur

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines))
        f.write("\n")
        f.write("\n".join(all_events))
        f.write("\n")

    logger.info("Merged %d subtitle files → %s (%d events)", len(subtitle_paths), output_path, len(all_events))
    return output_path


def _shift_dialogue_line(line: str, offset_sec: float) -> str:
    """Shift ASS Dialogue timestamps by offset_sec."""
    # Format: Dialogue: 0,H:MM:SS.cc,H:MM:SS.cc,Style,...
    parts = line.split(",", 3)
    if len(parts) < 4:
        return line
    try:
        start = _ass_to_seconds(parts[1])
        end = _ass_to_seconds(parts[2])
        new_start = _seconds_to_ass(start + offset_sec)
        new_end = _seconds_to_ass(end + offset_sec)
        return f"{parts[0]},{new_start},{new_end},{parts[3]}"
    except (ValueError, IndexError):
        return line


def _clamp_dialogue_to_boundary(line: str, block_end_sec: float) -> Optional[str]:
    """Clamp a Dialogue line so it does not exceed block_end_sec.

    Returns None if the event starts at or after block_end_sec (drop it).
    """
    parts = line.split(",", 3)
    if len(parts) < 4:
        return line
    try:
        start = _ass_to_seconds(parts[1])
        end = _ass_to_seconds(parts[2])
    except (ValueError, IndexError):
        return line
    if start >= block_end_sec - 0.01:
        return None  # event is entirely outside this block
    if end > block_end_sec:
        new_end = _seconds_to_ass(block_end_sec)
        return f"{parts[0]},{parts[1]},{new_end},{parts[3]}"
    return line


def _ass_to_seconds(timestamp: str) -> float:
    """Parse ASS timestamp H:MM:SS.cc to seconds."""
    ts = timestamp.strip()
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def build_merged_background(
    backgrounds: list[Path],
    durations: list[float],
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
) -> Path:
    """
    Build one continuous background video from per-block background segments.

    Each background[i] is trimmed to durations[i] seconds, scaled/cropped,
    then all are concatenated into one seamless video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(backgrounds) == 1:
        # Single background — just scale/crop and trim to total duration
        total = sum(durations)
        cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(backgrounds[0])]
        cmd += ["-t", f"{total:.3f}"]
        cmd += [
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,fps={fps}",
        ]
        cmd += ["-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22"]
        cmd += [str(output_path)]
        _run_ffmpeg(cmd, "build_merged_background")
        return output_path

    # Multiple backgrounds — trim each to its block duration, then concat
    segment_paths: list[Path] = []
    for i, (bg, dur) in enumerate(zip(backgrounds, durations)):
        seg_path = output_path.parent / f"{output_path.stem}_bgseg{i}.mp4"
        cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(bg)]
        cmd += ["-t", f"{dur:.3f}"]
        cmd += [
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,fps={fps}",
        ]
        cmd += ["-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22"]
        cmd += [str(seg_path)]
        _run_ffmpeg(cmd, f"bg_segment_{i}")
        segment_paths.append(seg_path)

    # Concat all segments
    list_file = output_path.parent / f"{output_path.stem}_bglist.txt"
    with open(list_file, "w") as f:
        for seg in segment_paths:
            f.write(f"file '{seg.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]
    _run_ffmpeg(cmd, "concat_bg_segments")

    # Cleanup segments
    list_file.unlink(missing_ok=True)
    for seg in segment_paths:
        seg.unlink(missing_ok=True)

    logger.info("Merged %d backgrounds → %s", len(backgrounds), output_path)
    return output_path


# _MAX_OVERLAYS is now set from profile config at module top


def _premix_pop_sounds(
    merged_audio_path: Path,
    overlay_start_times: list[float],
    pop_volume: float = 0.4,
) -> Path:
    """Pre-mix pop.mp3 into merged audio at each overlay start time.

    Returns path to the pre-mixed WAV file. This avoids N adelay filters
    in FFmpeg, saving hundreds of MB of RAM.
    """
    audio_data, sr = sf.read(str(merged_audio_path), dtype="float32")
    is_mono = audio_data.ndim == 1
    if is_mono:
        audio_data = audio_data[:, np.newaxis]

    # Load pop sound, resample if needed
    pop_data, pop_sr = sf.read(str(POP_SFX), dtype="float32")
    if pop_data.ndim == 1:
        pop_data = pop_data[:, np.newaxis]

    # Simple resample by repeating/skipping if sample rates differ
    if pop_sr != sr:
        ratio = sr / pop_sr
        indices = (np.arange(int(len(pop_data) * ratio)) / ratio).astype(int)
        indices = np.clip(indices, 0, len(pop_data) - 1)
        pop_data = pop_data[indices]

    # Match channels
    if pop_data.shape[1] < audio_data.shape[1]:
        pop_data = np.tile(pop_data, (1, audio_data.shape[1]))[:, :audio_data.shape[1]]
    elif pop_data.shape[1] > audio_data.shape[1]:
        pop_data = pop_data[:, :audio_data.shape[1]]

    pop_data = pop_data * pop_volume
    pop_len = len(pop_data)

    for start_sec in overlay_start_times:
        start_sample = int(start_sec * sr)
        end_sample = min(start_sample + pop_len, len(audio_data))
        paste_len = end_sample - start_sample
        if paste_len > 0:
            audio_data[start_sample:end_sample] += pop_data[:paste_len]

    # Clip to prevent distortion
    audio_data = np.clip(audio_data, -1.0, 1.0)

    if is_mono:
        audio_data = audio_data[:, 0]

    premixed_path = merged_audio_path.parent / f"{merged_audio_path.stem}_popmixed.wav"
    sf.write(str(premixed_path), audio_data, sr)
    logger.debug("Pre-mixed %d pop sounds into %s", len(overlay_start_times), premixed_path.name)

    # Free large numpy arrays immediately
    del audio_data, pop_data
    import gc
    gc.collect()

    return premixed_path


def render_single_pass(
    background_path: Path,
    merged_audio_path: Path,
    merged_subtitle_path: Path,
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    use_nvenc: bool = True,
    overlay_images: Optional[list[dict]] = None,
    image_mode: str = "popup",
) -> Path:
    """
    Render the entire video in one FFmpeg pass.

    One continuous looping background + one merged audio + one merged subtitle.
    Optional per-block overlay images with fade-in/fade-out.
    Pop sound is pre-mixed into audio (not via FFmpeg filters).

    image_mode:
      "popup"      — small centered overlay with floating sine animation + pop sound
      "background" — full-screen overlay with slow corner-to-corner pan, crossfade

    overlay_images: list of {"path": Path, "start_sec": float, "end_sec": float}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_info = sf.info(str(merged_audio_path))
    total_duration = audio_info.duration

    # --- Cap overlays to limit FFmpeg memory ---
    valid_overlays = [
        ovr for ovr in (overlay_images or [])
        if ovr.get("path") and Path(ovr["path"]).exists()
    ][:_MAX_OVERLAYS]

    # --- Pre-mix pop sounds into audio (popup mode only) ---
    actual_audio = merged_audio_path
    if image_mode == "popup" and POP_SFX.exists() and valid_overlays:
        try:
            pop_times = [float(ovr.get("start_sec", 0)) for ovr in valid_overlays]
            actual_audio = _premix_pop_sounds(merged_audio_path, pop_times)
        except Exception as exc:
            logger.warning("Pop sound pre-mix failed: %s. Using original audio.", exc)

    cmd = ["ffmpeg", "-y"]
    filters = []
    input_idx = 0

    # --- Input 0: Background ---
    if _is_image(background_path):
        cmd += ["-loop", "1", "-i", str(background_path)]
        filters.append(
            f"[0:v]scale=8000:-1,"
            f"zoompan=z='min(zoom+0.0005,1.3)'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d=1:s={width}x{height}:fps={fps},"
            f"setsar=1[bg]"
        )
    else:
        cmd += ["-stream_loop", "-1", "-i", str(background_path)]
        filters.append(
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,fps={fps}[bg]"
        )
    input_idx += 1

    # --- Input 1: Audio (pre-mixed with pop sounds) ---
    cmd += ["-i", str(actual_audio)]
    audio_input_idx = input_idx
    input_idx += 1

    # --- Subtitle burn-in ---
    sub_path_escaped = str(merged_subtitle_path).replace("\\", "/").replace(":", "\\:")
    filters.append(f"[bg]ass='{sub_path_escaped}'[subbed]")

    # --- Overlay images ---
    if valid_overlays:
        current_label = "subbed"

        if image_mode == "background":
            # Background mode: full-screen images with slow corner-to-corner pan
            bg_scale_w = int(width * 1.2)
            bg_scale_h = int(height * 1.2)
            x_pan_max = bg_scale_w - width
            y_pan_max = bg_scale_h - height

            for oi, ovr in enumerate(valid_overlays):
                start_sec = float(ovr.get("start_sec", 0))
                end_sec = float(ovr.get("end_sec", start_sec + 4.0))
                dur = max(end_sec - start_sec, 0.1)

                cmd += ["-loop", "1", "-i", str(Path(ovr["path"]))]
                ovr_input = input_idx
                input_idx += 1

                # Pick pan direction (cycle through presets)
                sx, sy, ex, ey = _BG_PAN_DIRS[oi % len(_BG_PAN_DIRS)]
                sx_px = int(sx * x_pan_max)
                sy_px = int(sy * y_pan_max)
                ex_px = int(ex * x_pan_max)
                ey_px = int(ey * y_pan_max)

                fade_dur = 0.3
                fade_in_start = start_sec
                fade_out_start = max(end_sec - fade_dur, fade_in_start + 0.1)

                ovr_label = f"ovr{oi}"
                out_label = f"v{oi}" if oi < len(valid_overlays) - 1 else "vout"

                # Scale to cover frame with 20% extra, crop to exact size
                filters.append(
                    f"[{ovr_input}:v]scale={bg_scale_w}:{bg_scale_h}"
                    f":force_original_aspect_ratio=increase,"
                    f"crop={bg_scale_w}:{bg_scale_h},format=rgba,"
                    f"fade=t=in:st={fade_in_start:.2f}:d={fade_dur}:alpha=1,"
                    f"fade=t=out:st={fade_out_start:.2f}:d={fade_dur}:alpha=1[{ovr_label}]"
                )

                # Animated pan: interpolate position from start to end corner
                prog = f"min(max((t-{start_sec:.2f})/{dur:.2f},0),1)"
                x_expr = f"-({sx_px}+({ex_px}-{sx_px})*{prog})"
                y_expr = f"-({sy_px}+({ey_px}-{sy_px})*{prog})"

                filters.append(
                    f"[{current_label}][{ovr_label}]overlay="
                    f"x='{x_expr}':y='{y_expr}':"
                    f"enable='between(t,{fade_in_start:.2f},{end_sec:.2f})'[{out_label}]"
                )
                current_label = out_label
        else:
            # Popup mode: centered overlay with gentle floating animation + pop sound
            ovr_width = int(width * 0.68)
            for oi, ovr in enumerate(valid_overlays):
                start_sec = float(ovr.get("start_sec", 0))
                end_sec = float(ovr.get("end_sec", start_sec + 3.0))

                cmd += ["-loop", "1", "-i", str(Path(ovr["path"]))]
                ovr_input = input_idx
                input_idx += 1

                fade_dur = 0.15
                fade_in_start = start_sec + 0.05
                fade_out_start = max(end_sec - 0.25, fade_in_start + fade_dur)
                overlay_end = fade_out_start + fade_dur

                ovr_label = f"ovr{oi}"
                out_label = f"v{oi}" if oi < len(valid_overlays) - 1 else "vout"

                filters.append(
                    f"[{ovr_input}:v]scale={ovr_width}:-1,format=rgba,"
                    f"fade=t=in:st={fade_in_start:.2f}:d={fade_dur}:alpha=1,"
                    f"fade=t=out:st={fade_out_start:.2f}:d={fade_dur}:alpha=1[{ovr_label}]"
                )
                # Floating: gentle sine wave on y (6px amplitude, ~4s cycle)
                float_y = f"(H-h)/2-40+6*sin((t-{start_sec:.2f})*1.5)"
                filters.append(
                    f"[{current_label}][{ovr_label}]overlay="
                    f"x=(W-w)/2:y='{float_y}':"
                    f"enable='between(t,{fade_in_start:.2f},{overlay_end:.2f})'[{out_label}]"
                )
                current_label = out_label
    else:
        filters.append("[subbed]copy[vout]")

    filter_complex = ";\n".join(filters)
    cmd += ["-filter_complex", filter_complex]
    cmd += ["-map", "[vout]", "-map", f"{audio_input_idx}:a"]
    cmd += ["-t", f"{total_duration:.3f}"]

    if use_nvenc:
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]
    cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += ["-movflags", "+faststart"]

    cmd += [str(output_path)]

    logger.info("Rendering single-pass video (%.1fs, %d overlays)...", total_duration, len(valid_overlays))
    _run_ffmpeg(cmd, "render_single_pass")

    # Clean up pre-mixed audio
    if actual_audio != merged_audio_path and actual_audio.exists():
        actual_audio.unlink(missing_ok=True)

    logger.info("Single-pass render done: %s", output_path)
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
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info("Mixing BGM...")
    _run_ffmpeg(cmd, "mix_bgm")
    logger.info("BGM mixed: %s", output_path)
    return output_path


def extract_audio_mp3(video_path: Path, output_path: Path) -> Path:
    """Extract the final audio track from a rendered video into an MP3 file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-c:a", "libmp3lame",
        "-q:a", "2",
        str(output_path),
    ]

    logger.info("Extracting MP3: %s", output_path.name)
    _run_ffmpeg(cmd, "extract_audio_mp3")
    logger.info("MP3 done: %s", output_path)
    return output_path


# ── Lab editor ──────────────────────────────────────────────────────────────

import random as _lab_random
from PIL import Image as _LabImage, ImageDraw as _LabImageDraw, ImageFilter as _LabImageFilter

_LAB_FONT_DIR   = Path(__file__).parent.parent / "assets" / "fonts"
_LAB_VIDEO_W    = 1080
_LAB_VIDEO_H    = 1920
_LAB_SIDE_MARGIN = 60
_LAB_VERT_MARGIN = 120
_LAB_CARD_W     = _LAB_VIDEO_W - _LAB_SIDE_MARGIN * 2   # 960
_LAB_CARD_H     = _LAB_VIDEO_H - _LAB_VERT_MARGIN * 2   # 1680
_LAB_PANEL_MARGIN = 84
_LAB_PANEL_W    = _LAB_VIDEO_W - _LAB_PANEL_MARGIN * 2  # 912
_LAB_PANEL_H    = int(_LAB_PANEL_W * 9 / 16)            # 513
_LAB_PANEL_TOP_Y    = 80
_LAB_PANEL_BOTTOM_Y = _LAB_VIDEO_H - _LAB_PANEL_TOP_Y - _LAB_PANEL_H  # 1327

_LAB_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Changa One,88,&H00FFFFFF,&H0054D8FF,&H000A0A0A,&H00000000,1,0,0,0,100,100,0.0,0,1,6.0,1.0,5,80,80,0,1
Style: Highlight,Changa One,88,&H0000F6FF,&H0000F6FF,&H00101010,&H00000000,1,0,0,0,100,100,0.0,0,1,7.0,1.0,5,80,80,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""

_LAB_XFADE = {
    "left":  "slideleft",
    "right": "slideright",
    "up":    "slideup",
    "down":  "slidedown",
}

_LAB_ASS_TAG_RE = re.compile(r"\{[^}]*\}")
_LAB_PUNCT_NO_SPACE_BEFORE = frozenset(",.!?;:%)]}\"'")
_LAB_PUNCT_NO_SPACE_AFTER  = frozenset("([{")


def _lab_ease(t: float) -> float:
    if t < 0.5:
        return 4 * t * t * t
    p = 2 * t - 2
    return 0.5 * p * p * p + 1


def _lab_cover_resize(img, w: int, h: int, bias_x: float = 0.5):
    src_w, src_h = img.size
    scale = max(w / src_w, h / src_h)
    resized = img.resize((int(src_w * scale), int(src_h * scale)), _LabImage.LANCZOS)
    max_left = max(resized.width - w, 0)
    left = int(max_left * bias_x)
    top  = max((resized.height - h) // 2, 0)
    return resized.crop((left, top, left + w, top + h))


def _lab_paste_panel(overlay, card, x: int, y: int) -> None:
    sw, sh = card.width + 40, card.height + 40
    shadow = _LabImage.new("RGBA", (sw, sh), (0, 0, 0, 0))
    _LabImageDraw.Draw(shadow).rounded_rectangle((0, 0, sw, sh), radius=44, fill=(0, 0, 0, 130))
    shadow = shadow.filter(_LabImageFilter.GaussianBlur(20))
    overlay.alpha_composite(shadow, (x - 20, y - 14))
    fw, fh = card.width + 16, card.height + 16
    frame = _LabImage.new("RGBA", (fw, fh), (248, 248, 248, 255))
    _LabImageDraw.Draw(frame).rounded_rectangle((0, 0, fw, fh), radius=32, fill=(248, 248, 248, 255))
    overlay.alpha_composite(frame, (x - 8, y - 8))
    overlay.alpha_composite(card.convert("RGBA"), (x, y))


def _lab_build_portrait_slide(img_path: Path, bias_x: float = 0.5):
    """Portrait-style card on a blurred background. Returns a PIL RGB Image."""
    with _LabImage.open(img_path).convert("RGB") as src:
        bg   = _lab_cover_resize(src, _LAB_VIDEO_W, _LAB_VIDEO_H, bias_x=0.5).filter(_LabImageFilter.GaussianBlur(26))
        card = _lab_cover_resize(src, _LAB_CARD_W, _LAB_CARD_H, bias_x=bias_x)
        base    = bg.convert("RGBA")
        overlay = _LabImage.new("RGBA", (_LAB_VIDEO_W, _LAB_VIDEO_H), (0, 0, 0, 0))
        _lab_paste_panel(overlay, card, _LAB_SIDE_MARGIN, _LAB_VERT_MARGIN)
        static  = _LabImage.alpha_composite(base, overlay).convert("RGB")
    return static


def _lab_build_dual_panel_slide(img_top_path: Path, img_bottom_path: Path, bias_top: float = 0.35, bias_bottom: float = 0.65):
    """Two landscape panels (top/bottom) with blurred background. Returns a PIL RGB Image."""
    with _LabImage.open(img_top_path).convert("RGB") as src_top:
        bg = _lab_cover_resize(src_top, _LAB_VIDEO_W, _LAB_VIDEO_H, bias_x=0.5).filter(_LabImageFilter.GaussianBlur(26))
        panel_top = _lab_cover_resize(src_top, _LAB_PANEL_W, _LAB_PANEL_H, bias_x=bias_top)
    with _LabImage.open(img_bottom_path).convert("RGB") as src_bot:
        panel_bot = _lab_cover_resize(src_bot, _LAB_PANEL_W, _LAB_PANEL_H, bias_x=bias_bottom)

    base    = bg.convert("RGBA")
    overlay = _LabImage.new("RGBA", (_LAB_VIDEO_W, _LAB_VIDEO_H), (0, 0, 0, 0))
    _lab_paste_panel(overlay, panel_top, _LAB_PANEL_MARGIN, _LAB_PANEL_TOP_Y)
    _lab_paste_panel(overlay, panel_bot, _LAB_PANEL_MARGIN, _LAB_PANEL_BOTTOM_Y)
    return _LabImage.alpha_composite(base, overlay).convert("RGB")


def _lab_merge_punctuation(words: list[dict]) -> list[dict]:
    """Collapse standalone punctuation tokens into the preceding word."""
    result: list[dict] = []
    punct_only = set(".,!?;:")
    for w in words:
        token = str(w.get("word", "")).strip()
        if token in punct_only and result:
            prev = dict(result[-1])
            prev["word"] = str(prev.get("word", "")).rstrip() + token
            prev["end"] = w["end"]
            result[-1] = prev
        else:
            result.append(w)
    return result


def _lab_chunk_words(words: list[dict], max_words: int = 6) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    current: list[dict] = []
    for word in words:
        current.append(word)
        token = str(word.get("word", "")).strip()
        if token.endswith((".", "!", "?")) or (token.endswith(",") and len(current) >= 3) or len(current) >= max_words:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)

    merged: list[list[dict]] = []
    for chunk in chunks:
        if len(chunk) == 1 and merged:
            merged[-1].extend(chunk)
        else:
            merged.append(chunk)

    if len(merged) > 1 and len(merged[0]) == 1:
        merged[1] = merged[0] + merged[1]
        merged = merged[1:]

    return merged


def _lab_join_ass_tokens(tokens: list[str]) -> str:
    """Join tokens with ASS-tag-aware punctuation spacing."""
    if not tokens:
        return ""
    result = [tokens[0]]
    for token in tokens[1:]:
        prev_clean = _LAB_ASS_TAG_RE.sub("", result[-1])
        next_clean = _LAB_ASS_TAG_RE.sub("", token)
        prev_last  = prev_clean[-1]  if prev_clean  else ""
        next_first = next_clean[0]   if next_clean  else ""
        if next_first in _LAB_PUNCT_NO_SPACE_BEFORE or prev_last in _LAB_PUNCT_NO_SPACE_AFTER:
            result.append(token)
        else:
            result.append(" " + token)
    return "".join(result)


def _lab_split_into_blocks(words: list[dict], target: int = 30) -> list[list[dict]]:
    """Split word list into blocks of ~target words, preferring sentence ends."""
    blocks: list[list[dict]] = []
    i = 0
    while i < len(words):
        end = min(i + target, len(words))
        best = end
        for j in range(end, max(i + target // 2, i + 1), -1):
            if str(words[j - 1].get("word", "")).strip().endswith((".", "!", "?")):
                best = j
                break
        blocks.append(words[i:best])
        i = best
    if len(blocks) > 1 and len(blocks[-1]) < 8:
        blocks[-2].extend(blocks[-1])
        blocks.pop()
    return blocks


def _lab_token_line_groups(chunk: list[dict]) -> list[list[str]]:
    plain   = _join_tokens([str(w["word"]) for w in chunk])
    wrapped = _wrap_caption_lines(plain, max_chars_per_line=18, max_lines=2)
    tokens  = [str(w["word"]) for w in chunk]
    if len(wrapped) <= 1 or len(tokens) <= 1:
        return [tokens]
    best: list[list[str]] | None = None
    best_score: tuple[float, float] | None = None
    for i in range(1, len(tokens)):
        left, right = tokens[:i], tokens[i:]
        score = (
            abs(len(_join_tokens(left)) - len(wrapped[0])) + abs(len(_join_tokens(right)) - len(wrapped[1])),
            abs(len(_join_tokens(left)) - len(_join_tokens(right))),
        )
        if best_score is None or score < best_score:
            best_score = score
            best = [left, right]
    return best or [tokens]


def _lab_highlight_text(line_groups: list[list[str]], active_idx: int) -> str:
    token_idx = 0
    lines: list[str] = []
    for group in line_groups:
        rendered: list[str] = []
        for token in group:
            if token_idx == active_idx:
                rendered.append(r"{\rHighlight}" + token + r"{\rDefault}")
            else:
                rendered.append(token)
            token_idx += 1
        lines.append(_lab_join_ass_tokens(rendered))
    return r"\N".join(lines)


def _lab_seconds_to_ass(s: float) -> str:
    total_cs = max(int(round(s * 100)), 0)
    cs = total_cs % 100
    total_s = total_cs // 100
    sec = total_s % 60
    m   = (total_s // 60) % 60
    h   = total_s // 3600
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"


def _lab_build_karaoke_ass(words: list[dict], audio_start: float) -> str:
    """Build a complete ASS file string (header + events). Timestamps are (word.start - audio_start)."""
    shifted = []
    for w in words:
        shifted.append({
            **w,
            "start": float(w["start"]) - audio_start,
            "end":   float(w["end"])   - audio_start,
        })

    raw: list[tuple[float, float, str]] = []
    chunks = _lab_chunk_words(_lab_merge_punctuation(shifted))
    for ci, chunk in enumerate(chunks):
        if not chunk:
            continue
        line_groups = _lab_token_line_groups(chunk)
        chunk_end = float(chunk[-1]["end"]) + 0.12
        if ci < len(chunks) - 1 and chunks[ci + 1]:
            nxt = float(chunks[ci + 1][0]["start"])
            chunk_end = min(chunk_end, nxt - 0.02)
        prev_end = float(chunk[0]["start"])
        for idx, word in enumerate(chunk):
            start = max(float(word["start"]), prev_end)
            if idx < len(chunk) - 1:
                end = max(float(chunk[idx + 1]["start"]), start + 0.04)
            else:
                end = max(chunk_end, start + 0.04)
            raw.append((start, end, _lab_highlight_text(line_groups, idx)))
            prev_end = end

    events: list[str] = []
    for i, (s, e, text) in enumerate(raw):
        if i < len(raw) - 1:
            e = min(e, raw[i + 1][0] - 0.001)
        if e < s + 0.001:
            continue
        events.append(f"Dialogue: 0,{_lab_seconds_to_ass(s)},{_lab_seconds_to_ass(e)},Default,,0,0,0,,{text}")

    return _LAB_ASS_HEADER + "\n" + "\n".join(events) + "\n"


def _lab_render_clip_video_only(slide: Path, duration: float, out: Path) -> None:
    """Render a static image loop as a video-only clip (no audio, no subtitles)."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(slide),
            "-t", f"{duration:.3f}",
            "-vf", "fps=30,format=yuv420p",
            "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out),
        ],
        check=True, capture_output=True,
    )


def _lab_escape_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", "\\:")


def _lab_apply_xfade(
    clip_paths: list[Path],
    block_durs: list[float],
    directions: list[str],
    xfade_dur: float,
    ass_path: Path,
    audio_path: Path,
    audio_start: float,
    out: Path,
) -> None:
    """Apply xfade transitions, burn global ASS, attach audio — one FFmpeg pass."""
    n = len(clip_paths)
    sub  = _lab_escape_path(ass_path)
    fdir = _lab_escape_path(_LAB_FONT_DIR)

    if n == 1:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(clip_paths[0]),
                "-ss", f"{audio_start:.3f}", "-i", str(audio_path),
                "-vf", f"fps=30,format=yuv420p,ass='{sub}':fontsdir='{fdir}'",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-shortest",
                str(out),
            ],
            check=True,
        )
        return

    # cut_times = cumulative end of each clip (before xfade reduction)
    cut_times: list[float] = []
    t = 0.0
    for d in block_durs[:-1]:
        t += d
        cut_times.append(t)

    filter_lines: list[str] = []
    prev_label = "[0:v]"
    accumulated_reduction = 0.0

    for i in range(n - 1):
        direction  = directions[i % len(directions)]
        xfade_name = _LAB_XFADE.get(direction, "fade")
        offset = cut_times[i] - xfade_dur / 2 - accumulated_reduction
        offset = max(offset, 0.01)
        accumulated_reduction += xfade_dur
        next_input = f"[{i + 1}:v]"
        mid_label  = f"[xf{i + 1}]"
        filter_lines.append(
            f"{prev_label}{next_input}xfade=transition={xfade_name}"
            f":duration={xfade_dur:.3f}:offset={offset:.3f}{mid_label}"
        )
        prev_label = mid_label

    filter_lines.append(
        f"{prev_label}fps=30,format=yuv420p,ass='{sub}':fontsdir='{fdir}'[vout]"
    )

    filter_complex = ";".join(filter_lines)
    audio_idx = n

    cmd = ["ffmpeg", "-y"]
    for p in clip_paths:
        cmd += ["-i", str(p)]
    cmd += ["-ss", f"{audio_start:.3f}", "-i", str(audio_path)]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", f"{audio_idx}:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def _lab_pick_bgm(bgm_dir: Path) -> Optional[Path]:
    """Randomly pick a .mp3 or .wav from the folder. Returns None if none found."""
    if not bgm_dir.exists():
        return None
    candidates = [p for p in bgm_dir.iterdir() if p.suffix.lower() in (".mp3", ".wav")]
    if not candidates:
        return None
    return _lab_random.choice(candidates)


def _lab_mix_bgm(video: Path, bgm: Path, out: Path, bgm_vol: float = 0.15) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video), "-i", str(bgm),
            "-filter_complex",
            f"[0:a][1:a]amix=inputs=2:duration=first:weights=1 {bgm_vol}[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            str(out),
        ],
        check=True,
    )
