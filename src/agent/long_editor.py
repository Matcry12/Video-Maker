"""Long-form video editor.

Composes a multi-chapter narration into a 10-15 min landscape video:
slow Ken Burns slides, ASS captions, chapter title overlays, mood BGM.

Single public entry: `compose(narration_wav, chapters, words, output_path, ...)`.
Reuses `src/editor.py` helpers where possible (BGM picker / mixer).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image as PILImage

from ..agent_config import load_agent_settings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
FONT_DIR = PROJECT_ROOT / "assets" / "fonts"
BGM_DIR = PROJECT_ROOT / "assets" / "audio" / "bgm"
BACKGROUND_VIDEO_DIR = PROJECT_ROOT / "assets" / "videos" / "backgrounds"
_BG_DEFAULT = BACKGROUND_VIDEO_DIR / "paper" / "brown.mp4"


def _pick_background_video(mood: str = "") -> Path:
    return _BG_DEFAULT


# ── ffmpeg helpers ──────────────────────────────────────────────────────────


def _run(cmd: list[str], label: str) -> None:
    logger.info("%s: %s", label, " ".join(cmd[:8]))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("%s failed (rc=%d):\nSTDERR: %s", label, res.returncode, res.stderr[-2000:])
        raise RuntimeError(f"{label} failed")


def _ffescape(p: Path) -> str:
    """Escape path for ffmpeg filter strings."""
    return str(p).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


# ── slide rendering ─────────────────────────────────────────────────────────


def _prepare_image(src: Path, dst: Path, width: int, height: int) -> None:
    """Cover-resize and save as JPEG so ffmpeg gets a clean rectangular input.
    Slight upscale margin is left for zoompan zoom-in."""
    margin = 1.10
    target_w = int(width * margin)
    target_h = int(height * margin)
    with PILImage.open(src) as im:
        im = im.convert("RGB")
        sw, sh = im.size
        scale = max(target_w / sw, target_h / sh)
        nw, nh = int(sw * scale), int(sh * scale)
        im = im.resize((nw, nh), PILImage.LANCZOS)
        # center-crop
        x0 = (nw - target_w) // 2
        y0 = (nh - target_h) // 2
        im = im.crop((x0, y0, x0 + target_w, y0 + target_h))
        im.save(dst, quality=92)


def _render_kenburns_clip(
    img: Path,
    duration: float,
    out: Path,
    width: int,
    height: int,
    fps: int,
    zoom_per_sec: float,
    max_zoom: float,
) -> None:
    """Render a single image as a video clip with a slow zoom-in (Ken Burns lite).

    Uses ffmpeg's zoompan filter. Output is yuv420p H.264 with no audio.
    """
    frames = max(int(round(duration * fps)), 1)
    z_step = zoom_per_sec / fps
    # Prepare a slightly oversized rectangular source so zoompan never
    # exposes black borders. zoompan reads from the input each frame, so we
    # can cap zoom at max_zoom to avoid blur on long durations.
    vf = (
        f"scale={int(width*1.10)}:{int(height*1.10)}:flags=lanczos,"
        f"zoompan="
        f"z='min(max(zoom\\,1.0)+{z_step:.6f}\\,{max_zoom:.4f})':"
        f"x='iw/2-(iw/zoom)/2':"
        f"y='ih/2-(ih/zoom)/2':"
        f"d={frames}:s={width}x{height}:fps={fps},"
        f"format=yuv420p"
    )
    _run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(img),
            "-t", f"{duration:.3f}",
            "-vf", vf,
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            str(out),
        ],
        label="kenburns",
    )


def _concat_clips(clips: list[Path], out: Path) -> None:
    """Concat clips with the demuxer (no re-encode). All clips must share codec/fps/size."""
    list_file = out.parent / f"{out.stem}_list.txt"
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in clips))
    _run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out),
        ],
        label="concat",
    )
    list_file.unlink(missing_ok=True)


# ── ASS subtitle building ───────────────────────────────────────────────────


def _ass_time(s: float) -> str:
    s = max(0.0, float(s))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - h * 3600 - m * 60
    return f"{h:d}:{m:02d}:{sec:05.2f}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")


_ASS_HEADER_TPL = """\
[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Changa One,{cap_size},&H00FFFFFF,&H0000F6FF,&H00101010,&H00000000,1,0,0,0,100,100,0.0,0,1,3.5,1.5,2,180,180,90,1
Style: ChapterTitle,Changa One,{title_size},&H00FFFFFF,&H0000F6FF,&H00101010,&H80000000,1,0,0,0,100,100,0.0,0,1,5.0,2.0,8,120,120,140,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _group_words_to_captions(
    words: list[dict],
    max_words: int = 9,
    min_words: int = 4,
    max_dur_sec: float = 4.5,
) -> list[dict]:
    """Group word timings into short caption chunks. Break on sentence punctuation."""
    captions: list[dict] = []
    cur: list[dict] = []
    for w in words:
        cur.append(w)
        token = str(w.get("word", "")).strip()
        ends_sentence = bool(re.search(r"[.!?…]['\"”’]?$", token))
        cur_dur = float(cur[-1]["end"]) - float(cur[0]["start"])
        if (
            (ends_sentence and len(cur) >= min_words)
            or len(cur) >= max_words
            or cur_dur >= max_dur_sec
        ):
            captions.append(cur)
            cur = []
    if cur:
        captions.append(cur)
    return captions


def _build_caption_dialogues(captions: list[list[dict]], video_h: int) -> list[str]:
    out: list[str] = []
    for chunk in captions:
        if not chunk:
            continue
        start = float(chunk[0]["start"])
        end = float(chunk[-1]["end"])
        if end <= start + 0.05:
            end = start + 0.6
        text = " ".join(str(w.get("word", "")).strip() for w in chunk if str(w.get("word", "")).strip())
        if not text:
            continue
        out.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,{_ass_escape(text)}"
        )
    return out


def _build_chapter_dialogues(chapters: list[dict], overlay_dur: float) -> list[str]:
    out: list[str] = []
    for i, ch in enumerate(chapters):
        title = str(ch.get("title", "")).strip()
        if not title:
            continue
        start = float(ch["start"])
        end = min(start + overlay_dur, float(ch["end"]))
        if end <= start + 0.1:
            continue
        # Fade in/out via ASS \fad tag.
        prefix = "{\\fad(400,600)}"
        label = f"Chapter {i+1}\\N{title}"
        out.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},ChapterTitle,,0,0,0,,{prefix}{_ass_escape(label)}"
        )
    return out


def _write_ass(
    path: Path,
    width: int,
    height: int,
    captions_words: list[dict],
    chapters: list[dict],
    chapter_overlay_sec: float,
) -> None:
    cap_size = max(36, int(height * 0.045))
    title_size = max(56, int(height * 0.072))
    header = _ASS_HEADER_TPL.format(w=width, h=height, cap_size=cap_size, title_size=title_size)
    cap_chunks = _group_words_to_captions(captions_words)
    cap_lines = _build_caption_dialogues(cap_chunks, height)
    title_lines = _build_chapter_dialogues(chapters, chapter_overlay_sec)
    path.write_text(header + "\n".join(cap_lines + title_lines) + "\n", encoding="utf-8")
    logger.info("ASS written: %s (%d caps, %d titles)", path, len(cap_lines), len(title_lines))


# ── BGM ─────────────────────────────────────────────────────────────────────


def _pick_bgm(mood: str) -> Optional[Path]:
    """Look in <bgm_dir>/<mood>/, fall back to flat dir."""
    import random
    if mood:
        sub = BGM_DIR / mood
        if sub.exists():
            cands = [p for p in sub.iterdir() if p.suffix.lower() in (".mp3", ".wav")]
            if cands:
                return random.choice(cands)
    if not BGM_DIR.exists():
        return None
    cands = [p for p in BGM_DIR.iterdir() if p.is_file() and p.suffix.lower() in (".mp3", ".wav")]
    return random.choice(cands) if cands else None


# ── final mux ──────────────────────────────────────────────────────────────


def _mux_with_overlays_and_audio(
    video_in: Path,
    audio_in: Path,
    ass_path: Path,
    out: Path,
    fps: int,
) -> None:
    """Burn ASS subtitles into the video and mux in the narration track."""
    vf = f"ass='{_ffescape(ass_path)}':fontsdir='{_ffescape(FONT_DIR)}'"
    _run(
        [
            "ffmpeg", "-y",
            "-i", str(video_in),
            "-i", str(audio_in),
            "-vf", vf,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(out),
        ],
        label="mux+subs",
    )


def _mix_bgm(video: Path, bgm: Path, out: Path, vol: float) -> None:
    _run(
        [
            "ffmpeg", "-y",
            "-i", str(video),
            "-stream_loop", "-1", "-i", str(bgm),
            "-filter_complex",
            f"[1:a]volume={vol}[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out),
        ],
        label="bgm_mix",
    )


# ── public API ─────────────────────────────────────────────────────────────


def compose(
    narration_wav: Path,
    chapters: list[dict[str, Any]],
    words: list[dict[str, Any]],
    output_path: Path,
    mood: str = "",
    emit: Optional[Callable[[dict], None]] = None,
) -> dict[str, Any]:
    """Build a long-form mp4 from per-chapter audio + image lists.

    chapters: each item must have:
        title (str), start (float, sec), end (float, sec), images (list[Path])
    words: list of {word, start, end} in the global narration timeline (already
        offset for inter-chapter gaps).
    output_path: final mp4 path.
    mood: BGM subfolder name; falls back to flat assets/audio/bgm/.

    Returns {"video_path": str, "audio_path": None, "duration_sec": float}.
    """
    def _emit(phase: str, message: str, **extra) -> None:
        if emit:
            try:
                emit({"phase": phase, "message": message, **extra})
            except Exception:
                pass

    settings = load_agent_settings()
    cfg = settings.get("long_form", {}) or {}
    res = cfg.get("resolution", {}) or {}
    width = int(res.get("width", 1920))
    height = int(res.get("height", 1080))
    fps = int(cfg.get("fps", 30))
    zoom_per_sec = float(cfg.get("ken_burns_zoom_per_sec", 0.0009))
    max_zoom = float(cfg.get("ken_burns_max_zoom", 1.18))
    min_img_sec = float(cfg.get("min_image_sec", 4.0))
    max_img_sec = float(cfg.get("max_image_sec", 7.0))
    chapter_overlay_sec = float(cfg.get("chapter_overlay_duration_sec", 2.5))
    bgm_volume = float(cfg.get("bgm_volume", 0.15))

    if not chapters:
        raise ValueError("compose: chapters is empty")
    if not narration_wav.exists():
        raise FileNotFoundError(f"narration_wav missing: {narration_wav}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="long_editor_"))
    logger.info("long_editor tmp: %s", tmp_dir)

    chapter_clips: list[Path] = []

    # 1. Per-chapter slideshow
    for ci, ch in enumerate(chapters):
        ch_dur = float(ch["end"]) - float(ch["start"])
        candidates = [Path(p) for p in ch.get("images", []) if Path(p).exists()]
        imgs: list[Path] = []
        for p in candidates:
            try:
                with PILImage.open(p) as _im:
                    _im.verify()
                imgs.append(p)
            except Exception as exc:
                logger.warning("skipping unreadable image %s: %s", p, exc)
        if not imgs:
            raise RuntimeError(f"chapter {ci} has no usable images")

        # Pick how many images to actually use given duration constraints
        max_n = max(1, int(ch_dur // min_img_sec))
        min_n = max(1, int(ch_dur // max_img_sec) + (1 if ch_dur % max_img_sec else 0))
        n = max(min_n, min(len(imgs), max_n))
        imgs = imgs[:n]
        per_img = ch_dur / n
        _emit("long_editor", f"Chapter {ci+1}: {n} images × {per_img:.2f}s")

        clip_paths: list[Path] = []
        for ii, img in enumerate(imgs):
            prepped = tmp_dir / f"c{ci}_img{ii}.jpg"
            _prepare_image(img, prepped, width, height)
            clip = tmp_dir / f"c{ci}_clip{ii}.mp4"
            _render_kenburns_clip(
                prepped, per_img, clip,
                width=width, height=height, fps=fps,
                zoom_per_sec=zoom_per_sec, max_zoom=max_zoom,
            )
            clip_paths.append(clip)
        ch_clip = tmp_dir / f"chapter_{ci}.mp4"
        if len(clip_paths) == 1:
            shutil.copy2(clip_paths[0], ch_clip)
        else:
            _concat_clips(clip_paths, ch_clip)
        chapter_clips.append(ch_clip)

    # 2. Concat all chapter videos
    _emit("long_editor", "Concatenating chapter videos")
    full_video = tmp_dir / "full_video.mp4"
    if len(chapter_clips) == 1:
        shutil.copy2(chapter_clips[0], full_video)
    else:
        _concat_clips(chapter_clips, full_video)

    # 3. Build ASS (captions + chapter overlays)
    _emit("long_editor", "Building subtitle + chapter overlays")
    ass_path = tmp_dir / "overlays.ass"
    _write_ass(ass_path, width, height, words, chapters, chapter_overlay_sec)

    # 4. Mux video + narration + ASS
    _emit("long_editor", "Muxing video + narration + subtitles")
    pre_bgm = tmp_dir / "pre_bgm.mp4"
    _mux_with_overlays_and_audio(full_video, narration_wav, ass_path, pre_bgm, fps=fps)

    # 5. BGM mix
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgm = _pick_bgm(mood or "")
    if bgm:
        _emit("long_editor", f"Mixing BGM ({bgm.name})")
        _mix_bgm(pre_bgm, bgm, output_path, vol=bgm_volume)
    else:
        _emit("long_editor", "No BGM found — copying without music")
        shutil.copy2(pre_bgm, output_path)

    duration = float(chapters[-1]["end"])
    return {
        "video_path": str(output_path),
        "audio_path": None,
        "duration_sec": duration,
    }


# ── card-on-blurred-bg composer (Shorts-lab style ported to 16:9) ───────────


def _cover_resize_pil(img, w: int, h: int, bias_x: float = 0.5):
    """Cover-fit + crop. Mirrors src.editor._lab_cover_resize."""
    src_w, src_h = img.size
    scale = max(w / src_w, h / src_h)
    resized = img.resize((int(src_w * scale), int(src_h * scale)), PILImage.LANCZOS)
    max_left = max(resized.width - w, 0)
    left = int(max_left * bias_x)
    top = max((resized.height - h) // 2, 0)
    return resized.crop((left, top, left + w, top + h))


def _contain_resize_card(img: "PILImage.Image", w: int, h: int) -> "PILImage.Image":
    """Contain-fit: entire image visible within (w, h), no cropping.
    Padding areas filled with a soft-blurred cover of the same image."""
    from PIL import ImageFilter as _ImageFilter
    src_w, src_h = img.size
    scale = min(w / src_w, h / src_h)
    nw, nh = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    resized = img.resize((nw, nh), PILImage.LANCZOS)
    bg = _cover_resize_pil(img, w, h, bias_x=0.5).filter(_ImageFilter.GaussianBlur(20))
    bg.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    return bg


def _paste_card_panel(overlay, card, x: int, y: int) -> None:
    """Paste card with soft drop shadow + light frame. Mirrors src.editor._lab_paste_panel."""
    from PIL import ImageDraw as _ImageDraw, ImageFilter as _ImageFilter
    sw, sh = card.width + 40, card.height + 40
    shadow = PILImage.new("RGBA", (sw, sh), (0, 0, 0, 0))
    _ImageDraw.Draw(shadow).rounded_rectangle((0, 0, sw, sh), radius=44, fill=(0, 0, 0, 130))
    shadow = shadow.filter(_ImageFilter.GaussianBlur(20))
    overlay.alpha_composite(shadow, (x - 20, y - 14))
    fw, fh = card.width + 16, card.height + 16
    frame = PILImage.new("RGBA", (fw, fh), (248, 248, 248, 255))
    _ImageDraw.Draw(frame).rounded_rectangle((0, 0, fw, fh), radius=32, fill=(248, 248, 248, 255))
    overlay.alpha_composite(frame, (x - 8, y - 8))
    overlay.alpha_composite(card.convert("RGBA"), (x, y))


def _build_landscape_card_slide(
    img_path: Path,
    video_w: int,
    video_h: int,
    card_w: int,
    card_h: int,
    bias_x: float = 0.5,
):
    """Landscape card slide: cover-resized image as a centered card sitting on a
    gaussian-blurred copy of the same image as the background. Returns PIL RGB."""
    from PIL import ImageFilter as _ImageFilter
    with PILImage.open(img_path) as src_raw:
        src = src_raw.convert("RGB")
        bg = _cover_resize_pil(src, video_w, video_h, bias_x=0.5).filter(_ImageFilter.GaussianBlur(28))
        card = _cover_resize_pil(src, card_w, card_h, bias_x=bias_x)
    base = bg.convert("RGBA")
    overlay = PILImage.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
    side_margin = (video_w - card_w) // 2
    top_margin = (video_h - card_h) // 2
    _paste_card_panel(overlay, card, side_margin, top_margin)
    return PILImage.alpha_composite(base, overlay).convert("RGB")


def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return 4.0 * t * t * t
    p = 2.0 * t - 2.0
    return 0.5 * p * p * p + 1.0


_CARD_MOTIONS = (
    "static",
)


def _viewport_for_motion(
    motion: str, p: float, src_w: int, src_h: int,
    card_w: int, card_h: int, zoom_strength: float = 0.18,
) -> tuple[int, int, int, int]:
    """Return the (vx, vy, vw, vh) crop window inside the oversized inner image.
    The crop is later resized back to (card_w, card_h)."""
    max_x = max(src_w - card_w, 0)
    max_y = max(src_h - card_h, 0)
    cx = max_x // 2
    cy = max_y // 2
    if motion == "pan_LR":
        return (int(max_x * p), cy, card_w, card_h)
    if motion == "pan_RL":
        return (int(max_x * (1.0 - p)), cy, card_w, card_h)
    if motion == "pan_TB":
        return (cx, int(max_y * p), card_w, card_h)
    if motion == "pan_BT":
        return (cx, int(max_y * (1.0 - p)), card_w, card_h)
    if motion == "pan_TL_BR":
        return (int(max_x * p), int(max_y * p), card_w, card_h)
    if motion == "pan_BR_TL":
        return (int(max_x * (1.0 - p)), int(max_y * (1.0 - p)), card_w, card_h)
    if motion == "zoom_in":
        s = 1.0 - zoom_strength * p
        vw = max(int(card_w * s), 8)
        vh = max(int(card_h * s), 8)
        return ((src_w - vw) // 2, (src_h - vh) // 2, vw, vh)
    if motion == "zoom_out":
        s = (1.0 - zoom_strength) + zoom_strength * p
        vw = max(int(card_w * s), 8)
        vh = max(int(card_h * s), 8)
        return ((src_w - vw) // 2, (src_h - vh) // 2, vw, vh)
    return (cx, cy, card_w, card_h)


def _build_static_card_layer(
    img_path: Path,
    video_w: int,
    video_h: int,
    card_w: int,
    card_h: int,
) -> tuple[np.ndarray, int, int]:
    """Build the per-slide static base: blurred bg + drop shadow + light frame.
    Card region is left untouched (filled per frame in compose).
    Returns (np.uint8 RGB layer, card_x, card_y)."""
    from PIL import ImageDraw as _ImageDraw, ImageFilter as _ImageFilter
    with PILImage.open(img_path) as src_raw:
        src = src_raw.convert("RGB")
        bg = _cover_resize_pil(src, video_w, video_h, bias_x=0.5).filter(_ImageFilter.GaussianBlur(28))
    base = bg.convert("RGBA")
    overlay = PILImage.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
    card_x = (video_w - card_w) // 2
    card_y = (video_h - card_h) // 2
    sw, sh = card_w + 40, card_h + 40
    shadow = PILImage.new("RGBA", (sw, sh), (0, 0, 0, 0))
    _ImageDraw.Draw(shadow).rounded_rectangle((0, 0, sw, sh), radius=44, fill=(0, 0, 0, 130))
    shadow = shadow.filter(_ImageFilter.GaussianBlur(20))
    overlay.alpha_composite(shadow, (card_x - 20, card_y - 14))
    fw, fh = card_w + 16, card_h + 16
    frame = PILImage.new("RGBA", (fw, fh), (248, 248, 248, 255))
    _ImageDraw.Draw(frame).rounded_rectangle((0, 0, fw, fh), radius=32, fill=(248, 248, 248, 255))
    overlay.alpha_composite(frame, (card_x - 8, card_y - 8))

    static_pil = PILImage.alpha_composite(base, overlay).convert("RGB")
    return np.array(static_pil), card_x, card_y


def _render_card_slide_clip(
    img_path: Path,
    duration: float,
    motion: str,
    out: Path,
    video_w: int,
    video_h: int,
    card_w: int,
    card_h: int,
    fps: int,
    bg_video: Path,
    inner_oversample: float = 1.0,  # unused, kept for API compat
    bias_x: float = 0.5,
) -> None:
    """Render one slide: contain-fit card overlay composited over looping background video.

    Step A — PIL renders card_overlay.png (RGBA, transparent background):
      - Drop shadow behind the border (blur=22, offset=(0,8), opacity=160)
      - Black 6px border rectangle (sharp corners)
      - Contain-fit card image pasted on top

    Step B — FFmpeg composites the overlay over the looping bg_video.
    """
    from PIL import ImageDraw as _ImageDraw, ImageFilter as _ImageFilter

    card_x = (video_w - card_w) // 2
    card_y = (video_h - card_h) // 2

    # ── Step A: build RGBA card overlay PNG ────────────────────────────────
    with PILImage.open(img_path) as src_raw:
        card_img = _contain_resize_card(src_raw.convert("RGB"), card_w, card_h)

    overlay = PILImage.new("RGBA", (video_w, video_h), (0, 0, 0, 0))

    border_px = 6
    fw = card_w + border_px * 2
    fh = card_h + border_px * 2
    bx = card_x - border_px
    by = card_y - border_px

    shadow_blur = 22
    shadow_offset = (0, 8)
    shadow_opacity = 160

    pad = shadow_blur * 2
    shadow_canvas = PILImage.new("RGBA", (fw + pad * 2, fh + pad * 2), (0, 0, 0, 0))
    _ImageDraw.Draw(shadow_canvas).rectangle(
        (pad, pad, pad + fw, pad + fh), fill=(0, 0, 0, shadow_opacity)
    )
    shadow_canvas = shadow_canvas.filter(_ImageFilter.GaussianBlur(shadow_blur))
    sx = bx - pad + shadow_offset[0]
    sy = by - pad + shadow_offset[1]
    overlay.alpha_composite(shadow_canvas, (sx, sy))

    border = PILImage.new("RGBA", (fw, fh), (0, 0, 0, 255))
    overlay.alpha_composite(border, (bx, by))

    overlay.alpha_composite(card_img.convert("RGBA"), (card_x, card_y))

    # Save overlay PNG to a temp file alongside the output
    overlay_png = out.with_suffix(".overlay.png")
    overlay.save(overlay_png, "PNG")

    # ── Step B: FFmpeg composite overlay over looping background ───────────
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-t", f"{duration:.3f}", "-i", str(bg_video),
        "-loop", "1", "-i", str(overlay_png),
        "-filter_complex", "[0:v][1:v]overlay=(W-w)/2:(H-h)/2:format=auto[out]",
        "-map", "[out]",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(out),
    ]
    _run(cmd, f"card_slide {out.name}")

    # Clean up temp overlay PNG
    try:
        overlay_png.unlink()
    except Exception:
        pass


# ── karaoke ASS (landscape variant of the Shorts lab style) ────────────────


def _long_token_line_groups(
    chunk: list[dict], max_chars_per_line: int = 30
) -> list[list[str]]:
    """Landscape variant of `_lab_token_line_groups` — same algo but wider lines."""
    from src.editor import _wrap_caption_lines, _join_tokens
    plain = _join_tokens([str(w["word"]) for w in chunk])
    wrapped = _wrap_caption_lines(plain, max_chars_per_line=max_chars_per_line, max_lines=2)
    tokens = [str(w["word"]) for w in chunk]
    if len(wrapped) <= 1 or len(tokens) <= 1:
        return [tokens]
    best: list[list[str]] | None = None
    best_score: tuple[float, float] | None = None
    for i in range(1, len(tokens)):
        left, right = tokens[:i], tokens[i:]
        score = (
            abs(len(_join_tokens(left)) - len(wrapped[0])) +
            abs(len(_join_tokens(right)) - len(wrapped[1])),
            abs(len(_join_tokens(left)) - len(_join_tokens(right))),
        )
        if best_score is None or score < best_score:
            best_score = score
            best = [left, right]
    return best or [tokens]


def _ass_rounded_rect_path(w: int, h: int, r: int) -> str:
    """ASS drawing path for a rounded rect centered on (0, 0).
    Uses cubic Bezier (b) for each corner. Coords in tenth-pixel units
    are unnecessary; we use whole pixels."""
    hw = w / 2.0
    hh = h / 2.0
    r = max(0, min(r, int(min(hw, hh))))
    k = r * 0.5523  # control-point offset for quarter-circle bezier
    parts = [
        f"m {-hw + r:.0f} {-hh:.0f}",
        f"l {hw - r:.0f} {-hh:.0f}",
        f"b {hw - r + k:.0f} {-hh:.0f} {hw:.0f} {-hh + r - k:.0f} {hw:.0f} {-hh + r:.0f}",
        f"l {hw:.0f} {hh - r:.0f}",
        f"b {hw:.0f} {hh - r + k:.0f} {hw - r + k:.0f} {hh:.0f} {hw - r:.0f} {hh:.0f}",
        f"l {-hw + r:.0f} {hh:.0f}",
        f"b {-hw + r - k:.0f} {hh:.0f} {-hw:.0f} {hh - r + k:.0f} {-hw:.0f} {hh - r:.0f}",
        f"l {-hw:.0f} {-hh + r:.0f}",
        f"b {-hw:.0f} {-hh + r - k:.0f} {-hw + r - k:.0f} {-hh:.0f} {-hw + r:.0f} {-hh:.0f}",
    ]
    return " ".join(parts)


def _build_long_karaoke_ass(
    path: Path,
    video_w: int,
    video_h: int,
    words: list[dict],
    chapters: list[dict],
    chapter_overlay_sec: float,
    sub_bar_bottom_margin: int,
    sub_bar_width_frac: float = 0.78,
    sub_bar_radius: int = 28,
    sub_bar_alpha: int = 178,
    sub_bar_pad_x: int = 60,
    sub_bar_pad_y: int = 24,
) -> tuple[int, int, int]:
    """Word-by-word highlight ASS for landscape long-form. The subtitle
    background is drawn as an ASS Drawing event positioned at the same
    y-anchor as the text — libass owns the position, so no pixel guessing.
    Bar width is uniform across the video (pre-measured from the longest
    caption line via PIL); height matches each caption's line count.
    Returns (n_caption_events, n_title_events, n_bar_events)."""
    from PIL import ImageFont
    from src.editor import (
        _lab_chunk_words, _lab_merge_punctuation,
        _lab_highlight_text, _lab_seconds_to_ass,
        _join_tokens,
    )

    cap_size = max(48, int(video_h * 0.058))
    title_size = max(56, int(video_h * 0.072))
    # Landscape chunking — fits more words per caption than the Shorts default.
    long_max_words = 12
    long_max_chars_per_line = 30

    # ── Pre-measure all caption lines with the same font we hand to libass ──
    font_path = FONT_DIR / "ChangaOne-Regular.ttf"
    font = ImageFont.truetype(str(font_path), cap_size)
    # Use the font's true line height (ascent + descent) to match what libass
    # actually renders. The glyph bbox alone is too short — it ignores leading.
    ascent, descent = font.getmetrics()
    line_h_px = ascent + descent
    chunks = _lab_chunk_words(_lab_merge_punctuation(words), max_words=long_max_words)
    measured: list[tuple[list[list[str]], float, float]] = []  # (line_groups, start, end)
    chunk_spans: list[tuple[float, float]] = []
    max_line_w = 0
    for ci, chunk in enumerate(chunks):
        if not chunk:
            continue
        line_groups = _long_token_line_groups(chunk, max_chars_per_line=long_max_chars_per_line)
        chunk_start = float(chunk[0]["start"])
        chunk_end = float(chunk[-1]["end"]) + 0.12
        if ci < len(chunks) - 1 and chunks[ci + 1]:
            nxt = float(chunks[ci + 1][0]["start"])
            chunk_end = min(chunk_end, nxt - 0.02)
        chunk_spans.append((chunk_start, chunk_end))
        measured.append((line_groups, chunk_start, chunk_end))
        for line in line_groups:
            plain = _join_tokens(line)
            bbox = font.getbbox(plain)
            w = int(bbox[2] - bbox[0])
            if w > max_line_w:
                max_line_w = w

    line_spacing = line_h_px  # no extra leading; libass renders compactly
    bar_w_max = int(video_w * sub_bar_width_frac)
    bar_w = min(max_line_w + 2 * sub_bar_pad_x, bar_w_max)
    bar_w = max(bar_w, 200)  # sanity floor
    # Lock bar height to the 2-line maximum so it never shifts between chunks.
    max_n_lines = 2
    fixed_bar_h = max_n_lines * line_spacing + 2 * sub_bar_pad_y
    fixed_bar_cy = video_h - sub_bar_bottom_margin - fixed_bar_h // 2
    fixed_text_margin_v = sub_bar_bottom_margin + sub_bar_pad_y

    # ASS BackColour with alpha. ASS alpha is inverted (0=opaque, 255=transparent).
    ass_alpha = max(0, min(255, 255 - int(sub_bar_alpha)))
    bar_color = f"&H{ass_alpha:02X}000000"  # &HAABBGGRR — black with chosen transparency

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_w}\n"
        f"PlayResY: {video_h}\n"
        "ScaledBorderAndShadow: yes\n"
        "WrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Changa One,{cap_size},&H00FFFFFF,&H0054D8FF,&H000A0A0A,"
        f"&H00000000,1,0,0,0,100,100,0.0,0,1,3.0,1.0,2,160,160,30,1\n"
        f"Style: Highlight,Changa One,{cap_size},&H0000F6FF,&H0000F6FF,&H00101010,"
        f"&H00000000,1,0,0,0,100,100,0.0,0,1,3.0,1.0,2,160,160,30,1\n"
        f"Style: SubBar,Arial,1,{bar_color},{bar_color},&H00000000,&H00000000,"
        "0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n"
        f"Style: ChapterTitle,Changa One,{title_size},&H00FFFFFF,&H0000F6FF,"
        "&H00101010,&H80000000,1,0,0,0,100,100,0.0,0,1,5.0,2.0,8,120,120,140,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    # ── Build bar + caption events. Bar position is fixed (never shifts). ──
    bar_events: list[str] = []
    cap_events: list[str] = []

    raw: list[tuple[float, float, str, int]] = []  # (start, end, text, n_lines)
    path_d = _ass_rounded_rect_path(bar_w, fixed_bar_h, sub_bar_radius)
    for (line_groups, chunk_start, chunk_end) in measured:
        n_lines = max(1, len(line_groups))
        bar_events.append(
            f"Dialogue: 0,{_lab_seconds_to_ass(chunk_start)},{_lab_seconds_to_ass(chunk_end)},"
            f"SubBar,,0,0,0,,{{\\pos({video_w // 2},{fixed_bar_cy})\\p1}}{path_d}{{\\p0}}"
        )
        raw.append(("__BAR__", chunk_start, chunk_end, n_lines, line_groups, fixed_text_margin_v))  # type: ignore

    # Now re-walk chunks for per-word events using the same timing logic as before.
    bar_iter = iter(raw)
    word_events_per_chunk: list[list[str]] = []
    for ci, chunk in enumerate(chunks):
        if not chunk:
            continue
        # Pull the matching bar tuple.
        _tag, chunk_start, chunk_end, n_lines, line_groups, text_margin_v = next(bar_iter)  # type: ignore
        prev_end = chunk_start
        events: list[tuple[float, float, str]] = []
        for idx, w in enumerate(chunk):
            start = max(float(w["start"]), prev_end)
            if idx < len(chunk) - 1:
                end = max(float(chunk[idx + 1]["start"]), start + 0.04)
            else:
                end = max(chunk_end, start + 0.04)
            events.append((start, end, _lab_highlight_text(line_groups, idx)))
            prev_end = end
        # Trim overlap with neighbour-of-next-event
        chunk_lines: list[str] = []
        for i, (s, e, text) in enumerate(events):
            if i < len(events) - 1:
                e = min(e, events[i + 1][0] - 0.001)
            if e < s + 0.001:
                continue
            # Per-event MarginV is the 8th Dialogue field.
            chunk_lines.append(
                f"Dialogue: 1,{_lab_seconds_to_ass(s)},{_lab_seconds_to_ass(e)},"
                f"Default,,0,0,{text_margin_v},,{text}"
            )
        word_events_per_chunk.append(chunk_lines)
        cap_events.extend(chunk_lines)

    title_events: list[str] = []
    for i, ch in enumerate(chapters):
        title = str(ch.get("title", "")).strip()
        if not title:
            continue
        start = float(ch["start"])
        end = min(start + chapter_overlay_sec, float(ch["end"]))
        if end <= start + 0.1:
            continue
        prefix = "{\\fad(400,600)}"
        label = f"Chapter {i + 1}\\N{title.replace('{', '').replace('}', '')}"
        title_events.append(
            f"Dialogue: 2,{_lab_seconds_to_ass(start)},{_lab_seconds_to_ass(end)},"
            f"ChapterTitle,,0,0,0,,{prefix}{label}"
        )

    path.write_text(
        header + "\n".join(bar_events + cap_events + title_events) + "\n",
        encoding="utf-8",
    )
    return (len(cap_events), len(title_events), len(bar_events))


def compose_card(
    narration_wav: Path,
    chapters: list[dict[str, Any]],
    words: list[dict[str, Any]],
    output_path: Path,
    mood: str = "",
    emit: Optional[Callable[[dict], None]] = None,
    bgm_volume_override: Optional[float] = None,
) -> dict[str, Any]:
    """Card-on-blurred-bg composer for long-form 16:9 video.

    Each image becomes a slide: gaussian-blurred copy of the image as background,
    a centered "card" with rounded shadow + light frame on top. Hard-cut between
    slides. Captions + chapter title burned via ASS, then narration + BGM muxed.

    Mirrors the Shorts lab style (`_lab_build_portrait_slide`) at 1920×1080.

    Same input contract as `compose()`.
    """
    def _emit(phase: str, message: str, **extra) -> None:
        if emit:
            try:
                emit({"phase": phase, "message": message, **extra})
            except Exception:
                pass

    settings = load_agent_settings()
    cfg = settings.get("long_form", {}) or {}
    res = cfg.get("resolution", {}) or {}
    width = int(res.get("width", 1920))
    height = int(res.get("height", 1080))
    fps = int(cfg.get("fps", 30))
    chapter_overlay_sec = float(cfg.get("chapter_overlay_duration_sec", 2.5))
    bgm_volume = float(bgm_volume_override) if bgm_volume_override is not None else float(cfg.get("bgm_volume", 0.15))
    card_w_frac = float(cfg.get("card_width_frac", 0.72))
    card_h_frac = float(cfg.get("card_height_frac", 0.72))
    min_img_sec = float(cfg.get("card_min_image_sec", 4.5))
    max_img_sec = float(cfg.get("card_max_image_sec", 7.5))

    card_w = int(width * card_w_frac)
    card_h = int(height * card_h_frac)

    # Subtitle bar — drawn as ASS Drawing event (libass owns position).
    # Width is uniform across the video (pre-measured from longest line).
    sub_bar_width_frac = float(cfg.get("subbar_width_frac", 0.78))
    sub_bar_radius = int(cfg.get("subbar_radius", 28))
    sub_bar_alpha = int(cfg.get("subbar_alpha", 220))  # 0-255, 220≈86 %
    sub_bar_bottom_margin = int(cfg.get("subbar_bottom_margin", 28))
    sub_bar_pad_x = int(cfg.get("subbar_pad_x", 24))
    sub_bar_pad_y = int(cfg.get("subbar_pad_y", 14))

    if not chapters:
        raise ValueError("compose_card: chapters is empty")
    if not narration_wav.exists():
        raise FileNotFoundError(f"narration_wav missing: {narration_wav}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="long_editor_card_"))
    logger.info("compose_card tmp: %s (card %dx%d)", tmp_dir, card_w, card_h)

    bias_cycle = [0.3, 0.5, 0.7, 0.4, 0.6, 0.3, 0.7, 0.5]
    chapter_clips: list[Path] = []
    motion_idx = 0

    for ci, ch in enumerate(chapters):
        ch_dur = float(ch["end"]) - float(ch["start"])
        candidates = [Path(p) for p in ch.get("images", []) if Path(p).exists()]
        imgs: list[Path] = []
        for p in candidates:
            try:
                with PILImage.open(p) as _im:
                    _im.verify()
                imgs.append(p)
            except Exception as exc:
                logger.warning("skipping unreadable image %s: %s", p, exc)
        if not imgs:
            raise RuntimeError(f"chapter {ci}: no usable images")

        # pick how many slide slots fit ch_dur within [min_img_sec, max_img_sec]
        max_n = max(1, int(ch_dur // min_img_sec))
        min_n = max(1, int(ch_dur // max_img_sec) + (1 if ch_dur % max_img_sec else 0))
        n = max(min_n, min(len(imgs), max_n))
        if n > len(imgs):
            imgs = (imgs * ((n // len(imgs)) + 1))[:n]
        else:
            imgs = imgs[:n]
        per = ch_dur / n
        _emit("long_editor", f"Chapter {ci+1}: {n} card slides × {per:.2f}s (image-in-box motion)")

        # Render each slide as its own clip with internal pan/zoom motion, then concat.
        bg_video = _pick_background_video(ch.get("mood", ""))
        clip_paths: list[Path] = []
        for ii, img in enumerate(imgs):
            motion = _CARD_MOTIONS[motion_idx % len(_CARD_MOTIONS)]
            motion_idx += 1
            clip = tmp_dir / f"c{ci}_clip{ii}.mp4"
            _render_card_slide_clip(
                img_path=img, duration=per, motion=motion, out=clip,
                video_w=width, video_h=height,
                card_w=card_w, card_h=card_h, fps=fps,
                bg_video=bg_video,
                bias_x=bias_cycle[ii % len(bias_cycle)],
            )
            clip_paths.append(clip)

        ch_clip = tmp_dir / f"chapter_{ci}.mp4"
        if len(clip_paths) == 1:
            shutil.copy2(clip_paths[0], ch_clip)
        else:
            _concat_clips(clip_paths, ch_clip)
        chapter_clips.append(ch_clip)

    _emit("long_editor", "Concatenating chapter videos")
    full_video = tmp_dir / "full_video.mp4"
    if len(chapter_clips) == 1:
        shutil.copy2(chapter_clips[0], full_video)
    else:
        _concat_clips(chapter_clips, full_video)

    _emit("long_editor", "Building karaoke captions + chapter overlays")
    ass_path = tmp_dir / "overlays.ass"
    n_caps, n_titles, n_bars = _build_long_karaoke_ass(
        ass_path, width, height, words, chapters, chapter_overlay_sec,
        sub_bar_bottom_margin=sub_bar_bottom_margin,
        sub_bar_width_frac=sub_bar_width_frac,
        sub_bar_radius=sub_bar_radius,
        sub_bar_alpha=sub_bar_alpha,
        sub_bar_pad_x=sub_bar_pad_x,
        sub_bar_pad_y=sub_bar_pad_y,
    )
    logger.info("ASS: %d caption events, %d bar events, %d chapter titles",
                n_caps, n_bars, n_titles)

    _emit("long_editor", "Muxing video + narration + subtitles")
    pre_bgm = tmp_dir / "pre_bgm.mp4"
    _mux_with_overlays_and_audio(full_video, narration_wav, ass_path, pre_bgm, fps=fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgm = _pick_bgm(mood or "")
    if bgm:
        _emit("long_editor", f"Mixing BGM ({bgm.name})")
        _mix_bgm(pre_bgm, bgm, output_path, vol=bgm_volume)
    else:
        _emit("long_editor", "No BGM found — copying without music")
        shutil.copy2(pre_bgm, output_path)

    duration = float(chapters[-1]["end"])
    return {
        "video_path": str(output_path),
        "audio_path": None,
        "duration_sec": duration,
    }
