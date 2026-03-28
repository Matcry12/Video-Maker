"""
Manager Module — Orchestrates the Content-to-Video pipeline.

Reads JSON scripts, maps roles to assets via profiles,
coordinates TTS synthesis and FFmpeg rendering.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from .tts import TTSEngine
from .editor import generate_ass, render_block_clip, concat_clips, mix_bgm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"
OUTPUT_DIR = PROJECT_ROOT / "output"


# --- Pydantic models for script validation ---

class Block(BaseModel):
    text: str
    voice: Optional[str] = None
    voice_rate: Optional[str] = None
    voice_pitch: Optional[str] = None
    voice_volume: Optional[str] = None
    background: Optional[str] = None  # image/video path for this block's background
    image: Optional[str] = None       # overlay image path (top-center popup)
    subtitle_preset: Optional[str] = None
    subtitle_style: Optional[str] = None
    subtitle_mode: Optional[str] = None  # standard | progressive
    subtitle_keywords: Optional[list[str]] = None


class Script(BaseModel):
    language: str = "vi-VN"
    background: Optional[str] = None  # main background (video/image) looping entire video
    voice_rate: Optional[str] = None
    voice_pitch: Optional[str] = None
    voice_volume: Optional[str] = None
    subtitle_preset: Optional[str] = None
    blocks: list[Block]


class GlobalAudio(BaseModel):
    bgm_folder: str = "assets/audio/bgm/"
    default_bgm_volume: float = 0.15


class TTSConfig(BaseModel):
    default_rate: str = "+0%"
    default_pitch: str = "+0Hz"
    default_volume: str = "+0%"


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
    presets: dict[str, SubtitlePreset] = Field(default_factory=_default_subtitle_presets)


class Profile(BaseModel):
    profile_name: str = "default"
    resolution: dict = Field(default_factory=lambda: {"width": 1080, "height": 1920})
    fps: int = 30
    default_voice: str = "NamMinh"
    default_background: str = "assets/videos/"
    global_audio: GlobalAudio = Field(default_factory=GlobalAudio)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    subtitle: SubtitleConfig = Field(default_factory=SubtitleConfig)


# --- Manager ---

class VideoManager:
    """Orchestrates script → TTS → subtitles → video render."""

    def __init__(
        self,
        profile_path: Optional[Path] = None,
        use_nvenc: bool = True,
    ):
        # Load profile
        if profile_path is None:
            profile_path = PROJECT_ROOT / "profiles" / "default.json"
        raw = json.loads(profile_path.read_text())
        self.profile = Profile(**raw)
        logger.info("Loaded profile: %s", self.profile.profile_name)

        self.use_nvenc = use_nvenc

        # Init TTS engine
        self.tts = TTSEngine()

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def close(self):
        self.tts.close()

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

    def _pick_bgm(self) -> Optional[Path]:
        """Pick a random background music track."""
        bgm_dir = PROJECT_ROOT / self.profile.global_audio.bgm_folder
        if not bgm_dir.exists():
            return None
        tracks = [f for f in bgm_dir.iterdir() if f.suffix.lower() in (".mp3", ".wav", ".ogg")]
        return random.choice(tracks) if tracks else None

    def _get_voice(self, block: Block) -> Optional[str]:
        """Resolve voice: block override > profile default."""
        return block.voice or self.profile.default_voice

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

    def _get_overlay(self, block: Block) -> Optional[Path]:
        """Resolve overlay image for a block."""
        if block.image:
            img = PROJECT_ROOT / block.image
            if img.exists():
                return img
            logger.warning("Overlay image not found: %s", img)
        return None

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

        return {
            "style": block.subtitle_style or preset.style,
            "highlight_style": preset.highlight_style,
            "subtitle_mode": block.subtitle_mode or preset.subtitle_mode,
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

        emit("tts", "Starting TTS synthesis...", 0, total_blocks)
        for i, block in enumerate(script.blocks):
            block_num = i + 1

            # Phase 1: TTS
            audio_path = TMP_DIR / f"{output_name}_block{i}.wav"
            voice = self._get_voice(block)
            tts_settings = self._resolve_tts_settings(block, script)
            emit("tts", f"TTS block {block_num}/{total_blocks}", block_num, total_blocks)
            result = self.tts.synthesize(
                text=block.text,
                output_path=audio_path,
                voice=voice,
                **tts_settings,
            )
            temp_files.append(audio_path)

            logger.info(
                "  Block %d/%d TTS done (%.1fs): %s...",
                i + 1, len(script.blocks), result["duration"], block.text[:30],
            )

            # Phase 2: Per-block subtitles (block-local timestamps)
            sub_path = TMP_DIR / f"{output_name}_block{i}.ass"
            subtitle_settings = self._resolve_subtitle_settings(block, script)
            emit("subtitle", f"Subtitle block {block_num}/{total_blocks}", block_num, total_blocks)
            generate_ass(
                result["words"],
                sub_path,
                **subtitle_settings,
            )
            temp_files.append(sub_path)

            # Phase 3: Resolve overlay
            overlay = self._get_overlay(block)

            # Phase 4: Render block clip
            clip_path = TMP_DIR / f"{output_name}_clip{i}.mp4"
            emit("rendering", f"Rendering block {block_num}/{total_blocks}", block_num, total_blocks)
            render_block_clip(
                background_path=background,
                audio_path=audio_path,
                subtitle_path=sub_path,
                output_path=clip_path,
                width=self.profile.resolution["width"],
                height=self.profile.resolution["height"],
                fps=self.profile.fps,
                overlay_image=overlay,
                background_offset_sec=background_offset_sec,
                use_nvenc=self.use_nvenc,
            )
            clip_paths.append(clip_path)
            temp_files.append(clip_path)
            background_offset_sec += max(float(result.get("duration", 0.0) or 0.0), 0.0)

        # Phase 5: Concatenate all clips
        concat_path = OUTPUT_DIR / f"{output_name}.mp4"
        bgm = self._pick_bgm()

        if bgm:
            # Concat to temp, then mix BGM
            concat_tmp = TMP_DIR / f"{output_name}_concat.mp4"
            emit("rendering", "Concatenating clips...", total_blocks, total_blocks)
            concat_clips(clip_paths, concat_tmp)
            temp_files.append(concat_tmp)
            emit("rendering", "Mixing background music...", total_blocks, total_blocks)
            mix_bgm(concat_tmp, bgm, concat_path, self.profile.global_audio.default_bgm_volume)
        else:
            emit("rendering", "Concatenating clips...", total_blocks, total_blocks)
            concat_clips(clip_paths, concat_path)

        # Cleanup temp files
        for f in temp_files:
            if f.exists():
                f.unlink(missing_ok=True)

        logger.info("Video complete: %s", concat_path)
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
