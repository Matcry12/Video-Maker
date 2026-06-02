"""Thumbnail generator for long-form videos.

Promoted from lab/longvideo/thumbnail.py.
Public entry: generate_thumbnail(video_path, title, run_dir, accent_word=None) -> Path
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).parent.parent
FONT_DIR = PROJECT_ROOT / "assets" / "fonts"
FONT_BOLD = FONT_DIR / "ChangaOne-Regular.ttf"
FONT_ITALIC = FONT_DIR / "ChangaOne-Italic.ttf"

THUMB_W, THUMB_H = 1280, 720
ACCENT = (255, 214, 64, 255)
ACCENT_RED = (231, 76, 60, 255)


# ── image helpers ────────────────────────────────────────────────────────────


def _font(size: int, italic: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_ITALIC if italic and FONT_ITALIC.exists() else FONT_BOLD
    try:
        return ImageFont.truetype(str(path), size=size)
    except Exception:
        return ImageFont.load_default()


def _cover_resize(img: Image.Image, w: int, h: int) -> Image.Image:
    sw, sh = img.size
    scale = max(w / sw, h / sh)
    r = img.resize((int(sw * scale), int(sh * scale)), Image.LANCZOS)
    left = (r.width - w) // 2
    top = (r.height - h) // 2
    return r.crop((left, top, left + w, top + h))


def _contain_resize(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    sw, sh = img.size
    scale = min(max_w / sw, max_h / sh)
    if scale >= 1.0:
        return img.copy()
    return img.resize((int(sw * scale), int(sh * scale)), Image.LANCZOS)


def _wrap(text: str, font: ImageFont.FreeTypeFont,
          draw: ImageDraw.ImageDraw, max_w: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        cur.append(w)
        bbox = draw.textbbox((0, 0), " ".join(cur), font=font)
        if bbox[2] - bbox[0] > max_w:
            cur.pop()
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _draw_outlined(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
                   font: ImageFont.FreeTypeFont,
                   fill: tuple[int, int, int, int],
                   stroke: int = 6,
                   stroke_fill: tuple[int, int, int, int] = (0, 0, 0, 255)) -> None:
    x, y = xy
    for dx in range(-stroke, stroke + 1):
        for dy in range(-stroke, stroke + 1):
            if dx * dx + dy * dy <= stroke * stroke and (dx or dy):
                draw.text((x + dx, y + dy), text, font=font, fill=stroke_fill)
    draw.text((x, y), text, font=font, fill=fill)


def _paste_card_with_shadow(canvas: Image.Image, card: Image.Image,
                             cx: int, cy: int) -> None:
    w, h = card.size
    shadow = Image.new("RGBA", (w + 80, h + 80), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (0, 0, w + 80, h + 80), radius=40, fill=(0, 0, 0, 180))
    shadow = shadow.filter(ImageFilter.GaussianBlur(28))
    canvas.alpha_composite(shadow, (cx - 40, cy - 25))
    border = Image.new("RGBA", (w + 16, h + 16), (255, 255, 255, 255))
    ImageDraw.Draw(border).rounded_rectangle(
        (0, 0, w + 16, h + 16), radius=22, fill=(255, 255, 255, 255))
    canvas.alpha_composite(border, (cx - 8, cy - 8))
    canvas.alpha_composite(card.convert("RGBA"), (cx, cy))


def _draw_title_block(canvas: Image.Image, title: str, accent_word: str,
                      x: int, y: int, max_w: int, max_h: int) -> None:
    draw = ImageDraw.Draw(canvas)
    chosen_size = 60
    chosen_lines: list[str] = [title]
    for size in range(140, 50, -4):
        f = _font(size)
        lines = _wrap(title, f, draw, max_w)
        total_h = len(lines) * int(size * 1.05)
        widest = max((draw.textbbox((0, 0), ln, font=f)[2] for ln in lines), default=0)
        if total_h <= max_h and widest <= max_w:
            chosen_size = size
            chosen_lines = lines
            break

    fnt = _font(chosen_size)
    line_h = int(chosen_size * 1.05)
    cy = y + (max_h - line_h * len(chosen_lines)) // 2
    accent_norm = accent_word.strip().lower()

    for line in chosen_lines:
        tokens = line.split(" ")
        widths = [draw.textbbox((0, 0), t, font=fnt)[2] - draw.textbbox((0, 0), t, font=fnt)[0]
                  for t in tokens]
        space_w = draw.textbbox((0, 0), " ", font=fnt)[2]
        total_w = sum(widths) + space_w * (len(tokens) - 1)
        cx = x + (max_w - total_w) // 2
        for i, tok in enumerate(tokens):
            stripped = tok.strip(".,!?:;\"'").lower()
            color = ACCENT if stripped == accent_norm else (255, 255, 255, 255)
            _draw_outlined(draw, (cx, cy), tok, fnt, fill=color, stroke=7)
            cx += widths[i] + space_w
        cy += line_h


def _make_16x9(frame: Image.Image, title: str, accent_word: str) -> Image.Image:
    fw, fh = frame.size
    if fw >= fh:
        return _make_16x9_fullbleed(frame, title, accent_word)

    # Portrait source: blurred bg + card on right + title on left
    bg = _cover_resize(frame, THUMB_W, THUMB_H).filter(ImageFilter.GaussianBlur(34))
    bg = ImageEnhance.Brightness(bg).enhance(0.45)
    bg = ImageEnhance.Color(bg).enhance(0.85)
    canvas = bg.convert("RGBA")

    card = _contain_resize(frame, 460, THUMB_H - 80)
    cx = THUMB_W - card.width - 60
    cy = (THUMB_H - card.height) // 2
    _paste_card_with_shadow(canvas, card, cx, cy)
    _draw_title_block(canvas, title, accent_word,
                      x=50, y=60, max_w=cx - 110, max_h=THUMB_H - 120)

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((50, THUMB_H - 40, 160, THUMB_H - 22),
                            radius=8, fill=ACCENT_RED)
    return canvas.convert("RGB")


def _make_16x9_fullbleed(frame: Image.Image, title: str, accent_word: str) -> Image.Image:
    fw, fh = frame.size
    # Crop bottom 25% to remove baked-in subtitle bars
    frame = frame.crop((0, 0, fw, int(fh * 0.75)))

    canvas = _cover_resize(frame, THUMB_W, THUMB_H).convert("RGBA")
    canvas.alpha_composite(Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 80)))

    panel_w = int(THUMB_W * 0.62)
    gradient = Image.new("RGBA", (panel_w, THUMB_H), (0, 0, 0, 0))
    for x in range(panel_w):
        alpha = int(230 * (1.0 - (x / panel_w) ** 1.3))
        for y in range(THUMB_H):
            gradient.putpixel((x, y), (5, 8, 18, alpha))
    canvas.alpha_composite(gradient, (0, 0))

    strip_h = 140
    sub_strip = Image.new("RGBA", (THUMB_W, strip_h), (0, 0, 0, 0))
    for y in range(strip_h):
        alpha = int(180 * (y / strip_h) ** 1.5)
        ImageDraw.Draw(sub_strip).line([(0, y), (THUMB_W, y)], fill=(0, 0, 0, alpha))
    canvas.alpha_composite(sub_strip, (0, THUMB_H - strip_h))

    _draw_title_block(canvas, title, accent_word,
                      x=48, y=60, max_w=panel_w - 80, max_h=THUMB_H - 120)

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((48, THUMB_H - 40, 160, THUMB_H - 22),
                            radius=8, fill=ACCENT_RED)
    return canvas.convert("RGB")


# ── image source ─────────────────────────────────────────────────────────────


def _find_background_image(run_dir: Optional[Path],
                            video_path: Optional[Path] = None,
                            seed: Optional[int] = None) -> Optional[Path]:
    rng = random.Random(seed)
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    pools: list[list[Path]] = []

    if run_dir and run_dir.exists():
        imgs = [p for p in run_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        if imgs:
            pools.append(imgs)

    cache_dir = PROJECT_ROOT / "tmp" / "cache" / "images"
    if cache_dir.exists():
        cached = [p for p in cache_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if cached:
            pools.append(cached)

    for pool in pools:
        if pool:
            return rng.choice(pool)
    return None


# ── youtube metadata txt ─────────────────────────────────────────────────────


def write_youtube_metadata(script_path: Path) -> Optional[Path]:
    """Write a plain-text youtube_metadata.txt next to script_path from its youtube field.
    Returns the output path, or None if no youtube field present."""
    import json
    data = json.loads(script_path.read_text(encoding="utf-8"))
    yt = data.get("youtube")
    if not yt:
        return None

    title       = yt.get("title", "")
    description = yt.get("description", "")
    chapters    = "\n".join(yt.get("chapters", []))
    chapters_note = yt.get("chapters_note", "")
    hashtags    = " ".join(yt.get("hashtags", []))
    tags        = ", ".join(yt.get("tags", []))

    txt = (
        f"TITLE\n{title}\n\n"
        f"DESCRIPTION\n{description}\n\n"
        f"CHAPTERS\n{chapters}\n"
        + (f"({chapters_note})\n" if chapters_note else "")
        + f"\nHASHTAGS\n{hashtags}\n\n"
        f"TAGS\n{tags}\n"
    )
    out = script_path.parent / "youtube_metadata.txt"
    out.write_text(txt, encoding="utf-8")
    return out


# ── public entry ─────────────────────────────────────────────────────────────


def generate_thumbnail(
    video_path: Path,
    title: str,
    run_dir: Optional[Path] = None,
    accent_word: Optional[str] = None,
    out_path: Optional[Path] = None,
    seed: Optional[int] = None,
) -> Path:
    """Generate a 1280×720 YouTube thumbnail next to the mp4.

    Picks the best available image from run_dir → image cache.
    Falls back to extracting a frame from the video if no images found.
    Returns the saved thumbnail path.
    """
    import subprocess, tempfile

    if accent_word is None:
        words = title.split()
        caps = [w.strip(".,!?:;\"'") for w in words if w.isupper() and len(w) > 2]
        accent_word = caps[0] if caps else words[0].strip(".,!?:;\"'")

    out = out_path or video_path.parent / f"{video_path.stem}_thumb.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)

    bg_path = _find_background_image(run_dir, video_path, seed=seed)

    if bg_path:
        with Image.open(bg_path).convert("RGB") as frame:
            img = _make_16x9(frame, title, accent_word)
    else:
        # Fallback: extract frame from video
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            frame_path = Path(tf.name)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", "4.0", "-i", str(video_path),
                 "-frames:v", "1", "-q:v", "2", str(frame_path)],
                check=True, capture_output=True,
            )
            with Image.open(frame_path).convert("RGB") as frame:
                img = _make_16x9(frame, title, accent_word)
        finally:
            frame_path.unlink(missing_ok=True)

    img.save(str(out), quality=97, optimize=True)
    return out
