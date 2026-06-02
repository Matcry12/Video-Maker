"""Compose a full-bleed b-roll Short: clips + TTS + karaoke captions + BGM.

Self-contained orchestration of ffmpeg + reused src helpers.

Pipeline:
  per beat -> build a background segment (clips cut at native speed, ~3s each,
              center-crop-filled to 9:16, concatenated to the beat's duration)
  concat all beat segments -> continuous background video (= narration length)
  burn karaoke ASS captions + attach narration audio
  mix BGM underneath
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

from . import clip_source, rerank
from src.agent_config import load_agent_settings

logger = logging.getLogger(__name__)

TARGET_W, TARGET_H = 1080, 1920
SUBCLIP_SEC = 3.0  # cut a new visual roughly every 3s for Shorts pacing


def build_beat_segment(
    beat_idx: int,
    queries: list[str],
    duration: float,
    work_dir: Path,
    layout: str = "fullbleed",
    bg_color: str = "black",
    used_ids: Optional[set] = None,
) -> Optional[Path]:
    """Build one beat's background video of length `duration` from stock clips.

    Fetches clips for the beat's queries, cuts ~SUBCLIP_SEC native-speed chunks,
    fits each (layout: fullbleed 9:16 or square-on-color), and concatenates until
    the beat duration is covered. Returns the segment path, or None if no footage.

    `used_ids` is a running set of Pexels ids already shown by earlier beats. Clips
    are picked from it FIRST excluded (cross-beat dedup), so the same footage never
    repeats across the video; each newly shown id is added to it in place. Only if a
    beat genuinely can't be filled from fresh clips does it fall back to re-using
    (with a varied start offset) so the visual still differs.
    """
    cfg = load_agent_settings().get("broll", {})
    rerank_on = cfg.get("rerank", True)
    if used_ids is None:
        used_ids = set()
    needed = duration
    segments: list[Path] = []
    seg_i = 0

    # Gather candidate clips across all queries for this beat (order preserved).
    candidates: list[dict] = []
    for q in queries:
        candidates.extend(clip_source.search_clips(q, per_page=15))
    if not candidates:
        logger.warning("beat %d: no clips for queries %s", beat_idx, queries)
        return None

    # De-dup by Pexels id within this beat, keep order.
    seen: set = set()
    uniq = []
    for c in candidates:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        uniq.append(c)

    # Cross-beat dedup: prefer clips not already used elsewhere; keep the rest as
    # a last-resort fallback pool so the beat can still be filled if needed.
    fresh = [c for c in uniq if c["id"] not in used_ids]
    fallback = [c for c in uniq if c["id"] in used_ids]
    pool = fresh + fallback
    if not fresh:
        logger.info("beat %d: all candidates already used; falling back", beat_idx)

    # --- SigLIP rerank (fresh clips only) -----------------------------------
    offset_by_id: dict = {}
    if rerank_on and fresh:
        reranked = rerank.rerank_pool(
            fresh, work_dir,
            max_candidates=cfg.get("rerank_candidates", 10),
            frames_per_clip=cfg.get("rerank_frames_per_clip", 4),
        )
        if reranked is not None:
            top = reranked[0]
            logger.info(
                "beat %d: rerank top score=%.3f query=%r",
                beat_idx, top.get("rerank_score", 0.0), top.get("query", ""),
            )
            rerank_floor = cfg.get("rerank_floor", 0.0)
            if rerank_floor > 0.0 and top.get("rerank_score", 0.0) < rerank_floor:
                logger.warning(
                    "beat %d: top rerank_score=%.3f below floor=%.2f; using anyway",
                    beat_idx, top.get("rerank_score", 0.0), rerank_floor,
                )
            pool = reranked + fallback
            offset_by_id = {c["id"]: c.get("best_offset", 0.0) for c in reranked}
    # ------------------------------------------------------------------------

    ci = 0
    guard = 0
    while needed > 0.15 and guard < 50:
        guard += 1
        cand = pool[ci % len(pool)]
        ci += 1
        src = clip_source.download_clip(cand)
        if not src:
            continue
        chunk = min(SUBCLIP_SEC, needed, max(1.0, cand["duration"]))
        # Use reranked best-frame offset on first use; varied formula on reuse.
        if cand["id"] not in used_ids and cand["id"] in offset_by_id:
            offset = max(0.0, min(offset_by_id[cand["id"]], cand["duration"] - chunk))
        else:
            # vary start offset so re-used clips don't show the same frames
            offset = min(max(0.0, cand["duration"] - chunk), (ci * 1.3) % max(0.1, cand["duration"]))
        seg_out = work_dir / f"beat{beat_idx}_seg{seg_i}.mp4"
        filled = clip_source.fill_segment(
            src, seg_out, chunk, start_offset=offset,
            layout=layout, bg_color=bg_color,
        )
        if not filled:
            continue
        used_ids.add(cand["id"])   # record globally so later beats skip it
        segments.append(filled)
        seg_i += 1
        needed -= chunk

    if not segments:
        return None

    # Concat this beat's sub-segments.
    beat_out = work_dir / f"beat{beat_idx}.mp4"
    if len(segments) == 1:
        # trim/pad single segment exactly to duration
        _exact_duration(segments[0], beat_out, duration)
    else:
        listed = work_dir / f"beat{beat_idx}_list.txt"
        listed.write_text("".join(f"file '{p.resolve()}'\n" for p in segments))
        tmp = work_dir / f"beat{beat_idx}_raw.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(listed), "-c", "copy", str(tmp)],
            check=True, capture_output=True,
        )
        _exact_duration(tmp, beat_out, duration)
    return beat_out


def _exact_duration(src: Path, out: Path, duration: float) -> None:
    """Re-encode `src` to exactly `duration` seconds (trim or freeze-pad last frame)."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-t", f"{duration:.3f}",
         "-vf", f"tpad=stop_mode=clone:stop_duration={duration:.3f},fps=30",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-t", f"{duration:.3f}", str(out)],
        check=True, capture_output=True,
    )


def concat_background(beat_videos: list[Path], work_dir: Path) -> Path:
    """Concatenate beat background videos into one continuous silent background."""
    listed = work_dir / "bg_list.txt"
    listed.write_text("".join(f"file '{p.resolve()}'\n" for p in beat_videos))
    bg = work_dir / "background.mp4"
    # re-encode (not copy) to guarantee uniform timebase across heterogeneous clips
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listed),
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", "30",
         str(bg)],
        check=True, capture_output=True,
    )
    return bg


def restyle_ass(
    ass_str: str,
    fontname: str,
    fontsize: int,
    alignment: int = 5,
    margin_v: int = 0,
    outline: int = 6,
    shadow: int = 1,
    margin_lr: int = 90,
) -> str:
    """Swap the Default/Highlight Style lines for a b-roll-specific look.

    The karaoke events reference styles by name and carry only inline colour/reset
    tags, so replacing the Style lines restyles everything (font, size, position)
    without re-implementing the karaoke chunking from src/editor.py.

    alignment uses ASS numpad codes: 2 = bottom-center, 5 = middle-center,
    8 = top-center.
    """
    def _style(name: str, primary: str) -> str:
        # Name,Font,Size,Primary,Secondary,Outline,Back,Bold,Italic,Underline,
        # StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,
        # Alignment,MarginL,MarginR,MarginV,Encoding
        return (
            f"Style: {name},{fontname},{fontsize},{primary},&H0054D8FF,"
            f"&H000A0A0A,&H00000000,1,0,0,0,100,100,0.0,0,1,"
            f"{outline}.0,{shadow}.0,{alignment},{margin_lr},{margin_lr},{margin_v},1"
        )

    out_lines = []
    for line in ass_str.splitlines():
        if line.startswith("Style: Default,"):
            out_lines.append(_style("Default", "&H00FFFFFF"))    # white
        elif line.startswith("Style: Highlight,"):
            out_lines.append(_style("Highlight", "&H0000F6FF"))  # yellow
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def burn_captions_and_audio(
    background: Path,
    ass_path: Path,
    audio_path: Path,
    out: Path,
    fontsdir: Path | None = None,
) -> Path:
    """Burn ASS karaoke captions onto the background and attach narration audio.

    `fontsdir` lets libass find a font that isn't installed system-wide (Changa
    One lives in assets/fonts, not in the OS font set).
    """
    def _esc(p) -> str:
        return str(p).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

    ass_filter = f"f='{_esc(ass_path)}'"
    if fontsdir is not None:
        ass_filter += f":fontsdir='{_esc(fontsdir)}'"

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(background),
         "-i", str(audio_path),
         "-filter_complex", f"[0:v]ass={ass_filter}[v]",
         "-map", "[v]", "-map", "1:a",
         "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest",
         str(out)],
        check=True, capture_output=True,
    )
    return out
