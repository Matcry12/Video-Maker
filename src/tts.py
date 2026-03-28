"""
TTS Module — Voice synthesis using Edge-TTS (Microsoft).

Fast, high-quality, supports Vietnamese with word-level timing.
No GPU required.
"""

import asyncio
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import edge_tts
import soundfile as sf

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "tmp" / "cache"

# Available Vietnamese voices
VOICES = {
    "HoaiMy": "vi-VN-HoaiMyNeural",    # Female
    "NamMinh": "vi-VN-NamMinhNeural",   # Male
}
DEFAULT_VOICE = "vi-VN-NamMinhNeural"
DEFAULT_RATE = "+0%"
DEFAULT_PITCH = "+0Hz"
DEFAULT_VOLUME = "+0%"


class TTSEngine:
    """Edge-TTS wrapper with caching and word-level timestamps."""

    def __init__(self, **kwargs):
        """No model loading needed — edge-tts is cloud-based."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Edge-TTS engine ready. Available voices:")
        for name, voice_id in VOICES.items():
            logger.info("  %s (%s)", name, voice_id)

    def close(self):
        """Nothing to release."""
        pass

    def _cache_key(
        self,
        text: str,
        voice: Optional[str],
        rate: str,
        pitch: str,
        volume: str,
    ) -> str:
        raw = json.dumps(
            {
                "text": text,
                "voice": voice,
                "rate": rate,
                "pitch": pitch,
                "volume": volume,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _resolve_voice(self, voice: Optional[str]) -> str:
        """Map short voice name to full edge-tts voice ID."""
        if not voice:
            return DEFAULT_VOICE
        # Check if it's already a full voice ID
        if voice.startswith("vi-VN-"):
            return voice
        # Map short name
        return VOICES.get(voice, DEFAULT_VOICE)

    def _normalize_percent(self, value: Optional[str], default: str, label: str) -> str:
        """
        Normalize values like 10, +10, 10%, +10% into Edge format (+10% / -5%).
        """
        if value is None:
            return default

        raw = str(value).strip()
        if raw == "":
            return default

        if raw.endswith("%"):
            number_str = raw[:-1].strip()
        else:
            number_str = raw

        try:
            number = float(number_str)
        except ValueError:
            logger.warning("Invalid %s '%s'; using default '%s'.", label, value, default)
            return default

        return f"{number:+g}%"

    def _normalize_pitch(self, value: Optional[str]) -> str:
        """
        Normalize values like 0, +20, 20Hz, +20Hz into Edge format (+20Hz / -10Hz).
        """
        if value is None:
            return DEFAULT_PITCH

        raw = str(value).strip()
        if raw == "":
            return DEFAULT_PITCH

        if raw.lower().endswith("hz"):
            number_str = raw[:-2].strip()
        else:
            number_str = raw

        try:
            number = float(number_str)
        except ValueError:
            logger.warning("Invalid pitch '%s'; using default '%s'.", value, DEFAULT_PITCH)
            return DEFAULT_PITCH

        return f"{number:+g}Hz"

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        volume: Optional[str] = None,
    ) -> dict:
        """
        Synthesize text to audio with word-level timestamps.

        Returns:
            dict with keys:
                audio_path: Path to the generated .mp3
                duration: float seconds
                words: list of {word, start, end}
        """
        voice_id = self._resolve_voice(voice)
        rate_value = self._normalize_percent(rate, DEFAULT_RATE, "rate")
        pitch_value = self._normalize_pitch(pitch)
        volume_value = self._normalize_percent(volume, DEFAULT_VOLUME, "volume")

        cache_key = self._cache_key(text, voice_id, rate_value, pitch_value, volume_value)
        cached_audio = CACHE_DIR / f"{cache_key}.mp3"
        cached_meta = CACHE_DIR / f"{cache_key}.json"

        # Return cached result if exists
        if cached_audio.exists() and cached_meta.exists():
            logger.info("Cache hit for '%s...'", text[:40])
            meta = json.loads(cached_meta.read_text())
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if cached_audio.resolve() != output_path.resolve():
                shutil.copy2(cached_audio, output_path)
            meta["audio_path"] = str(output_path)
            return meta

        # Synthesize
        logger.info(
            "Synthesizing: '%s...' (voice=%s, rate=%s, pitch=%s, volume=%s)",
            text[:40],
            voice_id,
            rate_value,
            pitch_value,
            volume_value,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        words = asyncio.run(
            self._generate(
                text=text,
                voice_id=voice_id,
                output_path=output_path,
                rate=rate_value,
                pitch=pitch_value,
                volume=volume_value,
            )
        )

        # Copy to cache
        shutil.copy2(output_path, cached_audio)

        # Get duration from the generated file
        info = sf.info(str(output_path))
        duration = info.duration

        # If edge-tts didn't provide word timing, use proportional spread
        if not words:
            words = self._proportional_spread(text.split(), 0.0, duration)

        meta = {
            "audio_path": str(output_path),
            "duration": round(duration, 3),
            "words": words,
            "voice": voice_id,
            "rate": rate_value,
            "pitch": pitch_value,
            "volume": volume_value,
        }
        cached_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        return meta

    async def _generate(
        self,
        text: str,
        voice_id: str,
        output_path: Path,
        rate: str,
        pitch: str,
        volume: str,
    ) -> list:
        """Run edge-tts and extract word timestamps from WordBoundary events."""
        communicate = edge_tts.Communicate(
            text,
            voice_id,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )

        words = []

        with open(output_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    offset_sec = chunk["offset"] / 10_000_000  # ticks to seconds
                    duration_sec = chunk["duration"] / 10_000_000
                    words.append({
                        "word": chunk["text"],
                        "start": round(offset_sec, 3),
                        "end": round(offset_sec + duration_sec, 3),
                    })

        return words

    def _proportional_spread(self, words: list, start: float, end: float) -> list:
        """Fallback: distribute words proportionally by character length."""
        if not words:
            return []
        total_duration = end - start
        weights = [max(len(w), 1) for w in words]
        total_weight = sum(weights)

        result = []
        cursor = start
        for w, weight in zip(words, weights):
            word_dur = total_duration * (weight / total_weight)
            result.append({
                "word": w,
                "start": round(cursor, 3),
                "end": round(cursor + word_dur, 3),
            })
            cursor += word_dur
        return result
