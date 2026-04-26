"""
Manager Module — Orchestrates the Content-to-Video pipeline.

Reads JSON scripts, maps roles to assets via profiles,
coordinates TTS synthesis and FFmpeg rendering.
"""

import json
import logging
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from .tts import TTSEngine, default_voice_for_language
from .agent_config import load_agent_settings

from PIL import Image as PILImage
from .editor import (
    generate_ass, render_block_clip, render_block_clip_fast, concat_clips,
    mix_bgm, extract_audio_mp3, concat_audio, merge_ass_subtitles,
    render_single_pass,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"
OUTPUT_DIR = PROJECT_ROOT / "output"


_DOWNSCALE_CACHE_DIR = TMP_DIR / "cache" / "resized"


def _nvenc_available() -> bool:
    """Probe whether NVENC hardware encoding is available."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def _downscale_image(path: Path, max_width: int = 734) -> Path:
    """Downscale image to max_width if larger. Saves as JPEG to reduce memory.

    Returns path to resized image (cached), or original if already small enough.
    """
    try:
        with PILImage.open(path) as img:
            # Always convert GIF/animated formats to static JPEG (animated GIFs break FFmpeg -loop 1)
            needs_convert = path.suffix.lower() in (".gif", ".webp")
            if img.width <= max_width and not needs_convert:
                return path
            _DOWNSCALE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            dest = _DOWNSCALE_CACHE_DIR / f"{path.stem}_{max_width}.jpg"
            if dest.exists() and dest.stat().st_size > 0:
                return dest
            if img.width > max_width:
                ratio = max_width / img.width
                new_h = int(img.height * ratio)
                resized = img.convert("RGB").resize((max_width, new_h), PILImage.LANCZOS)
            else:
                resized = img.convert("RGB")
            resized.save(dest, "JPEG", quality=85)
            return dest
    except Exception as exc:
        logger.warning("Image downscale failed for %s: %s", path, exc)
        return path


def _match_images_to_timeline(
    images: list,  # list[Path] or list[dict] with {"path": Path, "keyword": str}
    keywords: list[str],
) -> list:
    """Sort images to match image_keywords chronological order.

    Images with a matching keyword are placed at the position of that keyword.
    Images without a keyword match fill remaining slots.
    Returns list of Paths in timeline order.
    """
    if not images or not keywords:
        # No keywords to match — return paths in original order
        return [img if isinstance(img, Path) else Path(img.get("path", img)) for img in images]

    # Normalize: convert all to {"path": Path, "keyword": str}
    normalized = []
    for img in images:
        if isinstance(img, dict):
            normalized.append({"path": Path(img["path"]), "keyword": img.get("keyword", "")})
        else:
            normalized.append({"path": Path(img), "keyword": ""})

    # Build keyword slots (one per keyword, in order)
    slots = [None] * len(keywords)
    unmatched = []

    for img in normalized:
        if not img["keyword"]:
            unmatched.append(img["path"])
            continue

        # Find the best matching keyword slot
        best_idx = -1
        best_score = 0
        img_kw_lower = img["keyword"].lower()
        for ki, kw in enumerate(keywords):
            if slots[ki] is not None:
                continue  # slot already filled
            kw_lower = kw.lower()
            # Exact or substring match
            if img_kw_lower == kw_lower or img_kw_lower in kw_lower or kw_lower in img_kw_lower:
                score = len(set(img_kw_lower.split()) & set(kw_lower.split()))
                if score > best_score:
                    best_score = score
                    best_idx = ki

        if best_idx >= 0:
            slots[best_idx] = img["path"]
        else:
            unmatched.append(img["path"])

    # Fill empty slots with unmatched images
    result = []
    unmatched_iter = iter(unmatched)
    for slot in slots:
        if slot is not None:
            result.append(slot)
        else:
            try:
                result.append(next(unmatched_iter))
            except StopIteration:
                break

    # Append any remaining unmatched
    for remaining in unmatched_iter:
        result.append(remaining)

    return result


def _patch_ass_for_background(ass_path: Path) -> None:
    """Patch ASS subtitle styles for background mode: bigger font, centered slightly below middle."""
    try:
        import re as _re
        content = ass_path.read_text(encoding="utf-8")

        def _fix_style(m):
            fields = m.group(0).split(",")
            if len(fields) >= 23:
                # Increase font size by 20 (72→92, 74→94)
                try:
                    fields[2] = str(int(fields[2]) + 20)
                except ValueError:
                    pass
                # Keep alignment 2 (bottom-center), use high MarginV to push up
                # On 1920px height: MarginV=700 puts text slightly below center
                fields[18] = "2"
                fields[21] = "700"
            return ",".join(fields)

        content = _re.sub(
            r"^Style:\s*.+$",
            _fix_style,
            content,
            flags=_re.MULTILINE,
        )
        ass_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to patch ASS for background mode: %s", exc)


# --- Pydantic models for script validation ---

class Block(BaseModel):
    text: str
    voice: Optional[str] = None
    voice_rate: Optional[str] = None
    voice_pitch: Optional[str] = None
    voice_volume: Optional[str] = None
    background: Optional[str] = None  # image/video path for this block's background
    image: Optional[str | list[str]] = None  # overlay image path(s)
    subtitle_preset: Optional[str] = None
    subtitle_style: Optional[str] = None
    subtitle_mode: Optional[str] = None  # standard | progressive
    subtitle_alignment_mode: Optional[str] = None  # edge | corrected | forced
    subtitle_keywords: Optional[list[str]] = None
    image_keywords: Optional[list[str]] = None


class Script(BaseModel):
    language: str = "vi-VN"
    background: Optional[str] = None  # main background (video/image) looping entire video
    image_mode: str = "popup"  # "popup" (overlay with float) | "background" (full-screen pan)
    bgm_mood: str = ""  # "intense"|"calm"|"mystery"|"epic"|"emotional" — picks from mood subfolder
    voice_rate: Optional[str] = None
    voice_pitch: Optional[str] = None
    voice_volume: Optional[str] = None
    subtitle_preset: Optional[str] = None
    subtitle_alignment_mode: Optional[str] = None  # edge | corrected | forced
    blocks: list[Block]


class GlobalAudio(BaseModel):
    bgm_folder: str = "assets/audio/bgm/"
    default_bgm_volume: float = 0.15


class TTSConfig(BaseModel):
    default_rate: str = "+0%"
    default_pitch: str = "+0Hz"
    default_volume: str = "+0%"
    parallel_workers: int = 2


class SubtitlePreset(BaseModel):
    style: str = "Default"
    highlight_style: str = "Highlight"
    subtitle_mode: str = "standard"  # standard | progressive
    highlight_keywords: list[str] = Field(default_factory=list)
    max_chars_per_line: int = 28
    max_lines_per_caption: int = 2
    min_duration: float = 1.0
    max_duration: float = 2.5
    max_cps: float = 16.0
    pause_break_sec: float = 0.35


def _default_subtitle_presets() -> dict[str, SubtitlePreset]:
    return {
        "minimal": SubtitlePreset(
            style="Default",
            highlight_style="Highlight",
            subtitle_mode="standard",
            highlight_keywords=[],
            max_chars_per_line=30,
            max_lines_per_caption=2,
            min_duration=1.0,
            max_duration=2.7,
            max_cps=15.0,
            pause_break_sec=0.40,
        ),
        "energetic": SubtitlePreset(
            style="Default",
            highlight_style="Highlight",
            subtitle_mode="standard",
            highlight_keywords=[],
            max_chars_per_line=24,
            max_lines_per_caption=2,
            min_duration=0.8,
            max_duration=2.1,
            max_cps=17.0,
            pause_break_sec=0.30,
        ),
        "cinematic": SubtitlePreset(
            style="Default",
            highlight_style="Highlight",
            subtitle_mode="standard",
            highlight_keywords=[],
            max_chars_per_line=26,
            max_lines_per_caption=2,
            min_duration=1.2,
            max_duration=3.0,
            max_cps=14.0,
            pause_break_sec=0.45,
        ),
    }


class SubtitleConfig(BaseModel):
    default_preset: str = "minimal"
    default_alignment_mode: str = "corrected"
    presets: dict[str, SubtitlePreset] = Field(default_factory=_default_subtitle_presets)


class EditorConfig(BaseModel):
    min_img_sec: float = 6.0
    max_img_sec: float = 8.0


class Profile(BaseModel):
    profile_name: str = "default"
    resolution: dict = Field(default_factory=lambda: {"width": 1080, "height": 1920})
    fps: int = 30
    default_voice: str = "NamMinh"
    default_background: str = "assets/videos/"
    global_audio: GlobalAudio = Field(default_factory=GlobalAudio)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    subtitle: SubtitleConfig = Field(default_factory=SubtitleConfig)
    editor: EditorConfig = Field(default_factory=EditorConfig)


# --- Manager ---

class VideoManager:
    """Orchestrates script → TTS → subtitles → video render."""

    def __init__(
        self,
        profile_path: Optional[Path] = None,
        use_nvenc=None,
    ):
        # Load profile
        if profile_path is None:
            profile_path = PROJECT_ROOT / "profiles" / "default.json"
        raw = json.loads(profile_path.read_text())
        self.profile = Profile(**raw)
        logger.info("Loaded profile: %s", self.profile.profile_name)

        # Auto-detect NVENC if not explicitly set
        if use_nvenc is None:
            force_cpu = load_agent_settings().get("force_cpu_encode", False)
            use_nvenc = False if force_cpu else _nvenc_available()
        self.use_nvenc = use_nvenc
        logger.info("Encoder: %s", "h264_nvenc" if use_nvenc else "libx264")

        # Init TTS engine
        self.tts = TTSEngine()
        self._worker_state = threading.local()

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def close(self):
        self.tts.close()

    def _get_parallel_workers(self, total_blocks: int) -> int:
        """Choose a bounded worker count for block-local parallel processing."""
        configured = int(getattr(self.profile.tts, "parallel_workers", 2) or 1)
        env_override = os.getenv("VM_BLOCK_WORKERS")
        if env_override:
            try:
                configured = int(env_override)
            except ValueError:
                logger.warning("Invalid VM_BLOCK_WORKERS=%s; using profile/default value.", env_override)
        return max(1, min(total_blocks, configured))

    def _get_worker_tts(self) -> TTSEngine:
        """Create one TTS engine per worker thread to avoid shared async state."""
        engine = getattr(self._worker_state, "tts_engine", None)
        if engine is None:
            engine = TTSEngine()
            self._worker_state.tts_engine = engine
        return engine

    def _pick_asset(self, folder: str) -> Optional[Path]:
        """Pick a random video/image from an asset folder."""
        asset_dir = PROJECT_ROOT / folder
        if not asset_dir.exists():
            logger.warning("Asset folder not found: %s", asset_dir)
            return None
        files = [
            f for f in asset_dir.iterdir()
            if f.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov", ".jpg", ".png")
        ]
        if not files:
            logger.warning("No assets found in: %s", asset_dir)
            return None
        return random.choice(files)

    def _pick_bgm(self, bgm_mood: str = "") -> Optional[Path]:
        """Pick a background music track, preferring mood subdirectory if available.

        Lookup order:
        1. assets/audio/bgm/{bgm_mood}/ — mood-specific folder
        2. assets/audio/bgm/              — fallback to root folder
        """
        bgm_dir = PROJECT_ROOT / self.profile.global_audio.bgm_folder
        if not bgm_dir.exists():
            return None

        _AUDIO_EXTS = (".mp3", ".wav", ".ogg")

        # Try mood-specific subdirectory first
        if bgm_mood:
            mood_dir = bgm_dir / bgm_mood
            if mood_dir.is_dir():
                tracks = [f for f in mood_dir.iterdir() if f.suffix.lower() in _AUDIO_EXTS]
                if tracks:
                    logger.info("BGM: using mood '%s' (%d tracks)", bgm_mood, len(tracks))
                    return random.choice(tracks)
                logger.info("BGM: mood folder '%s' empty, falling back to root", bgm_mood)

        # Fallback: pick from root bgm folder (only files, not subdirs)
        tracks = [f for f in bgm_dir.iterdir() if f.is_file() and f.suffix.lower() in _AUDIO_EXTS]
        return random.choice(tracks) if tracks else None

    def _get_voice(self, block: Block, script: Script) -> Optional[str]:
        """Resolve voice: block override > language default > profile default."""
        if block.voice:
            return block.voice
        language_default = default_voice_for_language(script.language)
        return language_default or self.profile.default_voice

    def _get_background(self, script: Script) -> Optional[Path]:
        """
        Resolve one background for the whole timeline.

        Precedence:
        1) script.background
        2) first valid block.background (for backward compatibility)
        3) random asset from profile default folder
        """
        # Script-level main background
        if script.background:
            bg = PROJECT_ROOT / script.background
            if bg.exists():
                return bg
            logger.warning("Script background not found: %s", bg)
        # Backward compatibility: use first valid block background as global background
        for idx, block in enumerate(script.blocks):
            if not block.background:
                continue
            bg = PROJECT_ROOT / block.background
            if bg.exists():
                logger.info(
                    "Using block %d background as global background: %s",
                    idx + 1,
                    bg,
                )
                return bg
            logger.warning("Block background not found (block %d): %s", idx + 1, bg)
        # Profile default
        return self._pick_asset(self.profile.default_background)


    def _get_overlay(self, block: Block) -> list[Path]:
        """Resolve overlay image(s) for a block.

        block.image can be a single path string or a list of path strings.
        Returns list of existing Paths (may be empty).
        """
        if not block.image:
            return []
        # Normalise to list
        raw = block.image if isinstance(block.image, list) else [block.image]
        result: list[Path] = []
        for entry in raw:
            if not entry:
                continue
            img = PROJECT_ROOT / entry
            if img.exists():
                result.append(img)
            else:
                logger.warning("Overlay image not found: %s", img)
        return result

    def _resolve_subtitle_settings(self, block: Block, script: Script) -> dict:
        """
        Resolve subtitle settings with precedence:
        block override > script preset > profile default preset.
        """
        presets = self.profile.subtitle.presets or {}
        default_name = self.profile.subtitle.default_preset
        selected_name = block.subtitle_preset or script.subtitle_preset or default_name

        preset = presets.get(selected_name)
        if preset is None:
            logger.warning(
                "Subtitle preset '%s' not found. Falling back to '%s'.",
                selected_name,
                default_name,
            )
            preset = presets.get(default_name)

        if preset is None and presets:
            # Last-resort fallback to first available preset in profile.
            fallback_name, preset = next(iter(presets.items()))
            logger.warning("Using fallback subtitle preset '%s'.", fallback_name)

        if preset is None:
            preset = SubtitlePreset()

        if block.subtitle_keywords is None:
            keywords = list(preset.highlight_keywords)
        else:
            keywords = block.subtitle_keywords

        raw_mode = block.subtitle_mode or preset.subtitle_mode
        subtitle_mode = str(raw_mode or "standard").strip().lower()
        if subtitle_mode == "progressive":
            logger.warning(
                "Subtitle mode 'progressive' is deprecated for consistency. Falling back to 'standard'."
            )
            subtitle_mode = "standard"
        elif subtitle_mode != "standard":
            logger.warning(
                "Invalid subtitle_mode '%s'. Falling back to 'standard'.",
                subtitle_mode,
            )
            subtitle_mode = "standard"

        alignment_mode = (
            block.subtitle_alignment_mode
            if block.subtitle_alignment_mode is not None
            else script.subtitle_alignment_mode
            if script.subtitle_alignment_mode is not None
            else self.profile.subtitle.default_alignment_mode
        )
        alignment_mode = str(alignment_mode or "corrected").strip().lower()
        if alignment_mode not in {"edge", "corrected", "forced"}:
            logger.warning(
                "Invalid subtitle_alignment_mode '%s'. Falling back to 'corrected'.",
                alignment_mode,
            )
            alignment_mode = "corrected"

        return {
            "style": block.subtitle_style or preset.style,
            "highlight_style": preset.highlight_style,
            "subtitle_mode": subtitle_mode,
            "subtitle_alignment_mode": alignment_mode,
            "highlight_keywords": keywords,
            "max_chars_per_line": preset.max_chars_per_line,
            "max_lines_per_caption": preset.max_lines_per_caption,
            "min_duration": preset.min_duration,
            "max_duration": preset.max_duration,
            "max_cps": preset.max_cps,
            "pause_break_sec": preset.pause_break_sec,
        }

    def _resolve_tts_settings(self, block: Block, script: Script) -> dict:
        """
        Resolve TTS prosody settings with precedence:
        block override > script override > profile defaults.
        """
        return {
            "rate": (
                block.voice_rate
                if block.voice_rate is not None
                else script.voice_rate
                if script.voice_rate is not None
                else self.profile.tts.default_rate
            ),
            "pitch": (
                block.voice_pitch
                if block.voice_pitch is not None
                else script.voice_pitch
                if script.voice_pitch is not None
                else self.profile.tts.default_pitch
            ),
            "volume": (
                block.voice_volume
                if block.voice_volume is not None
                else script.voice_volume
                if script.voice_volume is not None
                else self.profile.tts.default_volume
            ),
        }

    def process_script(
        self,
        script_path: Path,
        output_name: Optional[str] = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> Path:
        """
        Process a JSON script file into a final video.
        Renders each block as a separate clip, then concatenates.
        """
        def emit(stage: str, message: str, current_block: Optional[int] = None, total_blocks: Optional[int] = None):
            if not progress_callback:
                return
            payload = {
                "stage": stage,
                "message": message,
            }
            if current_block is not None:
                payload["current_block"] = current_block
            if total_blocks is not None:
                payload["total_blocks"] = total_blocks
            progress_callback(payload)

        raw = json.loads(script_path.read_text())
        script = Script(**raw)
        total_blocks = len(script.blocks)
        logger.info("Processing script: %s (%d blocks)", script_path.name, total_blocks)

        if not output_name:
            output_name = script_path.stem

        clip_paths = []
        temp_files = []
        background = self._get_background(script)
        if not background:
            raise FileNotFoundError(
                "No background found. Add files to assets/videos/ or set 'background' in script."
            )
        background_offset_sec = 0.0
        max_workers = self._get_parallel_workers(total_blocks)

        def prepare_block(i: int, block: Block) -> dict:
            block_num = i + 1
            subtitle_settings = self._resolve_subtitle_settings(block, script)
            subtitle_alignment_mode = subtitle_settings.get("subtitle_alignment_mode", "corrected")
            ass_settings = {
                key: value for key, value in subtitle_settings.items() if key != "subtitle_alignment_mode"
            }
            audio_path = TMP_DIR / f"{output_name}_block{i}.wav"
            sub_path = TMP_DIR / f"{output_name}_block{i}.ass"
            voice = self._get_voice(block, script)
            tts_settings = self._resolve_tts_settings(block, script)
            tts_engine = self._get_worker_tts()
            import re as _re
            _tts_text = _re.sub(r'\s*\[[A-Z]\d+\]\s*', ' ', block.text).strip()
            result = tts_engine.synthesize(
                text=_tts_text,
                output_path=audio_path,
                voice=voice,
                alignment_mode=subtitle_alignment_mode,
                **tts_settings,
            )
            generate_ass(result["words"], sub_path, audio_duration=result["duration"], **ass_settings)

            # Background mode: bigger subtitles, positioned lower-center
            if script.image_mode == "background":
                _patch_ass_for_background(sub_path)

            logger.info(
                "  Block %d/%d prepared (%.1fs): %s...",
                block_num, len(script.blocks), result["duration"], block.text[:30],
            )
            return {
                "index": i,
                "audio_path": audio_path,
                "subtitle_path": sub_path,
                "overlay": self._get_overlay(block),
                "duration": max(float(result.get("duration", 0.0) or 0.0), 0.0),
            }

        logger.info("Using %d parallel block workers.", max_workers)

        emit("tts", "Starting TTS synthesis...", 0, total_blocks)
        prepared_blocks: list[Optional[dict]] = [None] * total_blocks
        prepared_count = 0
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vm-prepare") as executor:
            futures = {
                executor.submit(prepare_block, i, block): i
                for i, block in enumerate(script.blocks)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    prepared = future.result()
                except Exception as exc:
                    block_text = script.blocks[i].text[:50] if i < len(script.blocks) else "?"
                    logger.error(
                        "Block %d/%d TTS failed: %s. Text: '%s...'. Skipping block.",
                        i + 1, total_blocks, exc, block_text,
                    )
                    continue
                i = prepared["index"]
                prepared_blocks[i] = prepared
                temp_files.extend([prepared["audio_path"], prepared["subtitle_path"]])
                prepared_count += 1
                emit(
                    "subtitle",
                    f"Prepared block {prepared_count}/{total_blocks}",
                    prepared_count,
                    total_blocks,
                )

        # Filter out failed blocks (None entries) — render what we have
        failed_count = sum(1 for p in prepared_blocks if p is None)
        if failed_count > 0:
            logger.warning(
                "%d/%d blocks failed TTS. Rendering with %d successful blocks.",
                failed_count, total_blocks, total_blocks - failed_count,
            )
            prepared_blocks = [p for p in prepared_blocks if p is not None]
        if not prepared_blocks:
            raise RuntimeError(
                "All blocks failed TTS synthesis. Cannot render video. "
                "Check logs for per-block errors."
            )

        # Free TTS engines and thread-local data before render
        import gc
        gc.collect()

        # Phase 4: Block-by-block rendering
        # Render each block individually, then concat — keeps RAM low
        _MIN_IMG_SEC = self.profile.editor.min_img_sec
        _MAX_IMG_SEC = self.profile.editor.max_img_sec
        image_mode = script.image_mode
        if image_mode == "background":
            ovr_max_width = int(self.profile.resolution["width"] * 1.2)
        else:
            ovr_max_width = int(self.profile.resolution["width"] * 0.68)

        background_offset_sec = 0.0
        block_clip_paths: list[Path] = []

        render_count = len(prepared_blocks)
        for block_i, prepared in enumerate(prepared_blocks):
            emit("rendering", f"Rendering block {block_i + 1}/{render_count}...", block_i + 1, render_count)

            # Build overlay timeline for THIS block only (times relative to block start = 0)
            ovr_list = prepared.get("overlay") or []
            dur = prepared["duration"]

            # Background mode fallback: if this block has no images, borrow from neighbors
            if not ovr_list and image_mode == "background":
                for offset in range(1, render_count):
                    for neighbor_i in (block_i - offset, block_i + offset):
                        if 0 <= neighbor_i < render_count:
                            neighbor_ovr = prepared_blocks[neighbor_i].get("overlay") or []
                            if neighbor_ovr:
                                # Borrow images from neighbor — any images > 0 gaps
                                ovr_list = list(neighbor_ovr)  # copy to avoid mutation
                                logger.info(
                                    "Block %d has no images — borrowed %d from block %d",
                                    block_i, len(ovr_list), neighbor_i,
                                )
                                break
                    if ovr_list:
                        break
            # Sort images to match narration keyword order
            image_keywords = script.blocks[prepared["index"]].image_keywords or []
            ovr_list = _match_images_to_timeline(ovr_list, image_keywords)
            block_overlays: list[dict] = []

            if ovr_list:
                n_imgs = len(ovr_list)

                if image_mode == "background":
                    # Background mode: images ARE the background, must cover 100% of duration
                    # Each image gets 6-8 seconds, no gaps between them
                    # Use round() so 11.9s → 2 slots (5.95s each) instead of int() → 1 slot
                    avg_target = (_MIN_IMG_SEC + _MAX_IMG_SEC) / 2  # 7s
                    n_slots = max(1, round(dur / avg_target))
                    ovr_list = ovr_list[:n_slots]
                    n_imgs = len(ovr_list)
                    # Distribute time evenly, clamped to [6, 8]
                    base_time = dur / n_imgs
                    cursor = 0.0
                    for img_i, img_path in enumerate(ovr_list):
                        if img_path.exists():
                            small = _downscale_image(img_path, max_width=ovr_max_width)
                            is_last = (img_i == n_imgs - 1)
                            if is_last:
                                img_end = dur  # last image fills remaining time
                            else:
                                img_end = cursor + min(max(base_time, _MIN_IMG_SEC), _MAX_IMG_SEC)
                                img_end = min(img_end, dur)
                            block_overlays.append({
                                "path": small,
                                "start_sec": cursor,
                                "end_sec": img_end,
                            })
                            cursor = img_end  # NO gap — next image starts exactly where previous ends
                else:
                    # Popup mode: floating overlay images
                    sub_dur = dur / n_imgs if n_imgs > 0 else dur
                    if sub_dur < _MIN_IMG_SEC and n_imgs > 1:
                        n_slots = max(1, int(dur / _MIN_IMG_SEC))
                        ovr_list = ovr_list[:n_slots]
                        n_imgs = len(ovr_list)
                    cursor = 0.0
                    for img_i, img_path in enumerate(ovr_list):
                        if img_path.exists():
                            small = _downscale_image(img_path, max_width=ovr_max_width)
                            is_last = (img_i == n_imgs - 1)
                            img_end = dur if is_last else min(cursor + _MAX_IMG_SEC, dur)
                            block_overlays.append({
                                "path": small,
                                "start_sec": cursor,
                                "end_sec": img_end,
                            })
                            cursor = img_end

            clip_path = TMP_DIR / f"{output_name}_clip{block_i}.mp4"
            render_kwargs = dict(
                background_path=background,
                audio_path=prepared["audio_path"],
                subtitle_path=prepared["subtitle_path"],
                output_path=clip_path,
                width=self.profile.resolution["width"],
                height=self.profile.resolution["height"],
                fps=self.profile.fps,
                background_offset_sec=background_offset_sec,
                use_nvenc=self.use_nvenc,
                overlay_images=block_overlays if block_overlays else None,
                image_mode=image_mode,
            )
            if image_mode == "background" and block_overlays:
                try:
                    render_block_clip_fast(**render_kwargs)
                except Exception as exc:
                    logger.warning("Fast render failed: %s. Falling back to standard render.", exc)
                    render_block_clip(**render_kwargs)
            else:
                render_block_clip(**render_kwargs)
            block_clip_paths.append(clip_path)
            temp_files.append(clip_path)
            background_offset_sec += dur

            # Free this block's overlay data immediately
            del block_overlays, ovr_list
            gc.collect()

        # Free prepared_blocks — all data extracted
        del prepared_blocks
        gc.collect()

        # Phase 5: Concat all block clips into final video
        emit("rendering", "Concatenating blocks...", total_blocks, total_blocks)
        concat_path = OUTPUT_DIR / f"{output_name}.mp4"
        bgm = self._pick_bgm(bgm_mood=script.bgm_mood)

        if bgm:
            raw_concat = TMP_DIR / f"{output_name}_raw.mp4"
            concat_clips(block_clip_paths, raw_concat)
            temp_files.append(raw_concat)
            emit("rendering", "Mixing background music...", total_blocks, total_blocks)
            mix_bgm(raw_concat, bgm, concat_path, self.profile.global_audio.default_bgm_volume)
        else:
            concat_clips(block_clip_paths, concat_path)

        mp3_path = OUTPUT_DIR / f"{output_name}.mp3"
        emit("rendering", "Exporting MP3...", total_blocks, total_blocks)
        extract_audio_mp3(concat_path, mp3_path)

        # Cleanup temp files
        for f in temp_files:
            if f.exists():
                f.unlink(missing_ok=True)

        logger.info("Video complete: %s", concat_path)
        logger.info("Audio complete: %s", mp3_path)
        return concat_path


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m src.manager <script.json> [output_name]")
        print("Example: python -m src.manager json_scripts/3_tips.json my_video")
        sys.exit(1)

    script_path = Path(sys.argv[1])
    output_name = sys.argv[2] if len(sys.argv) > 2 else None

    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)

    manager = VideoManager()
    try:
        result = manager.process_script(script_path, output_name)
        print(f"\nDone! Video saved to: {result}")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
