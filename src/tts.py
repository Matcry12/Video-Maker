"""
TTS Module — Voice synthesis using Edge-TTS (Microsoft).

Fast, high-quality, supports Vietnamese with word-level timing.
No GPU required.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import edge_tts
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "tmp" / "cache"
SUBTITLE_DEBUG_DIR = Path(__file__).parent.parent / "tmp" / "subtitle_debug"
SUBTITLE_DEBUG_ENV = "VM_SUBTITLE_DEBUG"

# Available short-name voice aliases.
VOICES = {
    "HoaiMy": "vi-VN-HoaiMyNeural",    # Female
    "NamMinh": "vi-VN-NamMinhNeural",   # Male
    "Aria": "en-US-AriaNeural",         # Female
    "Guy": "en-US-GuyNeural",           # Male
}
DEFAULT_VOICE = "vi-VN-NamMinhNeural"
DEFAULT_VOICE_BY_LANGUAGE = {
    "vi": "vi-VN-NamMinhNeural",
    "en": "en-US-GuyNeural",
}
DEFAULT_RATE = "+0%"
DEFAULT_PITCH = "+0Hz"
DEFAULT_VOLUME = "+0%"
DEFAULT_ALIGNMENT_MODE = "corrected"
SUPPORTED_ALIGNMENT_MODES = {"edge", "corrected", "forced"}

TIMING_MIN_WORD_DURATION_SEC = 0.05
TIMING_MICRO_GAP_SEC = 0.015
TIMING_MAX_GAP_SEC = 0.24
TIMING_DRIFT_THRESHOLD_SEC = 0.16
TIMING_SCALE_MIN = 0.88
TIMING_SCALE_MAX = 1.18
TIMING_MAX_RAMP_SHIFT_SEC = 0.50
TIMING_TOTAL_RAMP_SHIFT_LIMIT_SEC = 0.80
TIMING_TAIL_MARGIN_SEC = 0.03
TIMING_EPS = 1e-6

# --- Parallel chunking constants ---
MIN_CHUNK_TEXT_LENGTH = 400        # Text shorter than this skips chunking
MAX_CHUNK_CHARS = 500              # Max characters per chunk
CHUNK_SEMAPHORE_LIMIT = 5          # Max concurrent edge-tts calls per synthesize()
_EDGE_TTS_GLOBAL_SEMAPHORE = threading.Semaphore(10)  # Global cap across all workers

# Sentence boundary regex — handles Vietnamese punctuation, avoids splitting on
# abbreviations (single uppercase letter + period), numbers with periods, ellipsis
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])'        # lookbehind: sentence-ending punctuation
    r'(?<!\b[A-Z]\.)'    # negative lookbehind: not single uppercase letter abbrev
    r'(?<!\d\.)'          # negative lookbehind: not digit-period (e.g. 2.5)
    r'\s+'                # require whitespace after punctuation
)


def _sanitize_for_tts(text: str) -> str:
    """Strip markdown, SSML, emojis, and non-speakable characters from text.

    edge-tts raises NoAudioReceived when fed markdown formatting, SSML tags,
    or certain Unicode characters. This function aggressively cleans the text.
    """
    import re as _re

    s = text

    # Strip markdown headings: ### Title -> Title
    s = _re.sub(r"^#{1,6}\s*", "", s, flags=_re.MULTILINE)

    # Strip bold/italic markers: **text** -> text, *text* -> text, __text__ -> text
    s = _re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", s)
    s = _re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", s)

    # Strip inline code: `code` -> code
    s = _re.sub(r"`([^`]*)`", r"\1", s)

    # Strip markdown links: [text](url) -> text
    s = _re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)

    # Strip HTML/SSML tags: <tag>text</tag> -> text
    s = _re.sub(r"<[^>]+>", "", s)

    # Strip bullet/list markers: - item, * item, 1. item
    s = _re.sub(r"^\s*[-*•]\s+", "", s, flags=_re.MULTILINE)
    s = _re.sub(r"^\s*\d+\.\s+", "", s, flags=_re.MULTILINE)

    # Normalize Unicode punctuation that edge-tts can't handle
    _UNICODE_REPLACEMENTS = {
        "\u2014": " - ",   # em-dash —
        "\u2013": " - ",   # en-dash –
        "\u2018": "'",     # left single quote '
        "\u2019": "'",     # right single quote '
        "\u201C": '"',     # left double quote "
        "\u201D": '"',     # right double quote "
        "\u2026": "...",   # ellipsis …
        "\u00A0": " ",     # non-breaking space
        "\u200B": "",      # zero-width space
        "\u200C": "",      # zero-width non-joiner
        "\u200D": "",      # zero-width joiner
        "\uFEFF": "",      # BOM
        "\u00AD": "",      # soft hyphen
        "\u2022": "",      # bullet •
        "\u2023": "",      # triangular bullet
        "\u25AA": "",      # small black square
        "\u25AB": "",      # small white square
        "\u00B7": "",      # middle dot ·
    }
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        s = s.replace(char, replacement)

    # Strip emojis and misc symbols (Unicode blocks)
    s = _re.sub(
        r"[\U0001F600-\U0001F64F"  # emoticons
        r"\U0001F300-\U0001F5FF"   # misc symbols & pictographs
        r"\U0001F680-\U0001F6FF"   # transport & map symbols
        r"\U0001F1E0-\U0001F1FF"   # flags
        r"\U00002702-\U000027B0"   # dingbats
        r"\U0000FE00-\U0000FE0F"   # variation selectors
        r"\U00002300-\U000023FF"   # misc technical
        r"\U00002B50-\U00002B55"   # stars
        r"\U0000200D"              # zero width joiner
        r"\U00010000-\U0010FFFF"   # all supplementary planes (catches remaining emojis)
        r"]+", "", s
    )

    # Strip any remaining control characters (C0/C1 except newline/tab)
    s = _re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", s)

    # Collapse multiple spaces/newlines
    s = _re.sub(r"\s+", " ", s).strip()

    return s


class TTSEngine:
    """Edge-TTS wrapper with caching and word-level timestamps."""

    def __init__(self, **kwargs):
        """No model loading needed — edge-tts is cloud-based."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.subtitle_debug_enabled = _truthy_env(os.getenv(SUBTITLE_DEBUG_ENV))
        if self.subtitle_debug_enabled:
            SUBTITLE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Edge-TTS engine ready. Available voices:")
        for name, voice_id in VOICES.items():
            logger.info("  %s (%s)", name, voice_id)
        logger.info(
            "Subtitle debug dump: %s",
            "enabled" if self.subtitle_debug_enabled else "disabled",
        )

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
        if voice.endswith("Neural") and "-" in voice:
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
        alignment_mode: Optional[str] = None,
    ) -> dict:
        """
        Synthesize text to audio with word-level timestamps.

        Returns:
            dict with keys:
                audio_path: Path to the generated .mp3
                duration: float seconds
                words: list of {word, start, end}
        """
        # Sanitize text to prevent edge-tts NoAudioReceived errors
        original_text = text
        text = _sanitize_for_tts(text)
        if not text:
            raise ValueError(
                f"Text is empty after sanitization — nothing to synthesize. "
                f"Original: '{original_text[:100]}'"
            )
        if text != original_text:
            logger.debug("TTS sanitized text: '%s...' → '%s...'", original_text[:60], text[:60])

        voice_id = self._resolve_voice(voice)
        rate_value = self._normalize_percent(rate, DEFAULT_RATE, "rate")
        pitch_value = self._normalize_pitch(pitch)
        volume_value = self._normalize_percent(volume, DEFAULT_VOLUME, "volume")
        requested_alignment_mode, effective_alignment_mode, fallback_reason = self._resolve_alignment_mode(
            alignment_mode
        )
        if fallback_reason:
            logger.warning(
                "Subtitle alignment mode fallback: requested=%s effective=%s reason=%s",
                requested_alignment_mode,
                effective_alignment_mode,
                fallback_reason,
            )

        cache_key = self._cache_key(text, voice_id, rate_value, pitch_value, volume_value)
        cached_audio = CACHE_DIR / f"{cache_key}.mp3"
        cached_meta = CACHE_DIR / f"{cache_key}.json"

        # --- Check for parallel chunking opportunity ---
        chunks = self._split_into_chunks(text)
        use_chunking = len(chunks) > 1

        # Return cached result if exists (full-text cache — applies to both paths)
        if cached_audio.exists() and cached_meta.exists():
            logger.info("Cache hit for '%s...'", text[:40])
            cached = json.loads(cached_meta.read_text())
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if cached_audio.resolve() != output_path.resolve():
                shutil.copy2(cached_audio, output_path)

            duration = float(cached.get("duration") or 0.0)
            raw_words = cached.get("raw_words")
            if raw_words is None:
                raw_words = cached.get("words") or cached.get("corrected_words") or []
                if effective_alignment_mode == "edge":
                    fallback_reason = fallback_reason or "edge_mode_unavailable_legacy_cache"
                    logger.warning(
                        "Subtitle alignment mode fallback: requested=%s effective=%s reason=%s",
                        requested_alignment_mode,
                        effective_alignment_mode,
                        fallback_reason,
                    )

            raw_words = self._coerce_words(raw_words)
            if not raw_words:
                raw_words = self._coerce_words(self._proportional_spread(text.split(), 0.0, duration))

            if effective_alignment_mode == "forced":
                from .whisper_align import align_audio
                forced_words = align_audio(
                    output_path,
                    text,
                    language=self._language_hint_from_voice(voice_id),
                )
                if forced_words:
                    raw_words = forced_words

            corrected_words, timing_correction, raw_timing_stats, corrected_timing_stats = (
                self._postprocess_word_timestamps(
                    words=raw_words,
                    duration=duration,
                )
            )
            words, timing_stats = self._pick_alignment_words_and_stats(
                effective_alignment_mode=effective_alignment_mode,
                words=raw_words,
                corrected_words=corrected_words,
                raw_timing_stats=raw_timing_stats,
                corrected_timing_stats=corrected_timing_stats,
            )

            self._log_timing_stats(cache_key, timing_stats, cache_hit=True)
            if effective_alignment_mode == "corrected":
                self._log_timing_correction(cache_key, timing_correction, cache_hit=True)

            selected_timing_correction = timing_correction if effective_alignment_mode == "corrected" else {}
            meta = {
                "audio_path": str(output_path),
                "duration": round(duration, 3),
                "raw_words": raw_words,
                "corrected_words": corrected_words,
                "words": words,
                "voice": cached.get("voice", voice_id),
                "rate": cached.get("rate", rate_value),
                "pitch": cached.get("pitch", pitch_value),
                "volume": cached.get("volume", volume_value),
                "timing_stats_raw": raw_timing_stats,
                "timing_stats_corrected": corrected_timing_stats,
                "timing_stats": timing_stats,
                "timing_correction": selected_timing_correction,
                "alignment_mode_requested": requested_alignment_mode,
                "alignment_mode_effective": effective_alignment_mode,
                "alignment_mode_fallback_reason": fallback_reason,
            }
            self._maybe_dump_subtitle_debug(
                cache_key=cache_key,
                text=text,
                output_path=output_path,
                words=words,
                timing_stats=timing_stats,
                voice=voice_id,
                rate=rate_value,
                pitch=pitch_value,
                volume=volume_value,
                duration=duration,
                cache_hit=True,
                raw_words=raw_words,
                raw_timing_stats=raw_timing_stats,
                timing_correction=selected_timing_correction,
                alignment_mode_requested=requested_alignment_mode,
                alignment_mode_effective=effective_alignment_mode,
                alignment_mode_fallback_reason=fallback_reason,
            )
            return meta

        # Synthesize
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if use_chunking:
            # --- PARALLEL CHUNKING PATH ---
            logger.info(
                "Parallel chunking: %d chunks for '%s...' (voice=%s, rate=%s)",
                len(chunks), text[:40], voice_id, rate_value,
            )
            t_start = time.monotonic()

            try:
                chunk_results = asyncio.run(
                    self._generate_chunks(
                        chunks=chunks,
                        voice_id=voice_id,
                        rate=rate_value,
                        pitch=pitch_value,
                        volume=volume_value,
                    )
                )
                duration = self._merge_chunk_audio(chunk_results, output_path)
                raw_words = self._merge_chunk_words(chunk_results)

                # Cleanup temp chunk files
                for cr in chunk_results:
                    chunk_path = Path(cr["audio_path"])
                    if chunk_path.exists() and "chunks" in str(chunk_path):
                        try:
                            chunk_path.unlink()
                        except OSError:
                            pass

                cache_hits = sum(1 for cr in chunk_results if cr.get("cache_hit"))
                elapsed = time.monotonic() - t_start
                logger.info(
                    "Parallel chunking done: %d chunks in %.1fs (cache hits: %d/%d)",
                    len(chunks), elapsed, cache_hits, len(chunks),
                )
            except Exception as exc:
                # Fallback to single-pass synthesis on any chunk error
                logger.warning("Parallel chunking failed: %s. Falling back to single-pass.", exc)
                use_chunking = False

        if not use_chunking:
            # --- SINGLE-PASS PATH (original) ---
            logger.info(
                "Synthesizing: '%s...' (voice=%s, rate=%s, pitch=%s, volume=%s)",
                text[:40],
                voice_id,
                rate_value,
                pitch_value,
                volume_value,
            )

            raw_words = asyncio.run(
                self._generate(
                    text=text,
                    voice_id=voice_id,
                    output_path=output_path,
                    rate=rate_value,
                    pitch=pitch_value,
                    volume=volume_value,
                )
            )

            # Get duration from the generated file
            info = sf.info(str(output_path))
            duration = info.duration

            # If edge-tts didn't provide word timing, use proportional spread
            if not raw_words:
                raw_words = self._proportional_spread(text.split(), 0.0, duration)

        raw_words = self._coerce_words(raw_words)

        # Copy to full-text cache
        shutil.copy2(output_path, cached_audio)

        if effective_alignment_mode == "forced":
            from .whisper_align import align_audio
            forced_words = align_audio(
                output_path,
                text,
                language=self._language_hint_from_voice(voice_id),
            )
            if forced_words:
                raw_words = forced_words

        corrected_words, timing_correction, raw_timing_stats, corrected_timing_stats = (
            self._postprocess_word_timestamps(
                words=raw_words,
                duration=duration,
            )
        )
        words, timing_stats = self._pick_alignment_words_and_stats(
            effective_alignment_mode=effective_alignment_mode,
            words=raw_words,
            corrected_words=corrected_words,
            raw_timing_stats=raw_timing_stats,
            corrected_timing_stats=corrected_timing_stats,
        )
        self._log_timing_stats(cache_key, timing_stats, cache_hit=False)
        if effective_alignment_mode == "corrected":
            self._log_timing_correction(cache_key, timing_correction, cache_hit=False)

        cache_meta = {
            "duration": round(duration, 3),
            "raw_words": raw_words,
            "corrected_words": corrected_words,
            "voice": voice_id,
            "rate": rate_value,
            "pitch": pitch_value,
            "volume": volume_value,
            "timing_stats_raw": raw_timing_stats,
            "timing_stats_corrected": corrected_timing_stats,
            "timing_correction": timing_correction,
        }
        cached_meta.write_text(json.dumps(cache_meta, ensure_ascii=False, indent=2))

        selected_timing_correction = timing_correction if effective_alignment_mode == "corrected" else {}
        meta = {
            "audio_path": str(output_path),
            **cache_meta,
            "words": words,
            "timing_stats": timing_stats,
            "timing_correction": selected_timing_correction,
            "alignment_mode_requested": requested_alignment_mode,
            "alignment_mode_effective": effective_alignment_mode,
            "alignment_mode_fallback_reason": fallback_reason,
        }
        self._maybe_dump_subtitle_debug(
            cache_key=cache_key,
            text=text,
            output_path=output_path,
            words=words,
            timing_stats=timing_stats,
            voice=voice_id,
            rate=rate_value,
            pitch=pitch_value,
            volume=volume_value,
            duration=duration,
            cache_hit=False,
            raw_words=raw_words,
            raw_timing_stats=raw_timing_stats,
            timing_correction=selected_timing_correction,
            alignment_mode_requested=requested_alignment_mode,
            alignment_mode_effective=effective_alignment_mode,
            alignment_mode_fallback_reason=fallback_reason,
        )
        return meta

    def _resolve_alignment_mode(self, alignment_mode: Optional[str]) -> tuple[str, str, Optional[str]]:
        """Resolve requested subtitle alignment mode and fallback strategy."""
        requested = str(alignment_mode or "").strip().lower()
        if not requested:
            requested = DEFAULT_ALIGNMENT_MODE

        if requested not in SUPPORTED_ALIGNMENT_MODES:
            return requested, DEFAULT_ALIGNMENT_MODE, "invalid_mode"
        if requested == "forced":
            return requested, "forced", None
        return requested, requested, None

    def _language_hint_from_voice(self, voice_id: str) -> Optional[str]:
        """Extract language hint from voice ID, e.g. 'vi-VN-NamMinhNeural' -> 'vi-VN'."""
        if not voice_id:
            return None
        parts = voice_id.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return None

    def _pick_alignment_words_and_stats(
        self,
        *,
        effective_alignment_mode: str,
        words: list[dict],
        corrected_words: list[dict],
        raw_timing_stats: dict,
        corrected_timing_stats: dict,
    ) -> tuple[list[dict], dict]:
        """Pick output timestamps/statistics according to effective alignment mode."""
        if effective_alignment_mode == "edge":
            return words, raw_timing_stats
        if effective_alignment_mode == "forced":
            return words, raw_timing_stats
        return corrected_words, corrected_timing_stats

    async def _generate(
        self,
        text: str,
        voice_id: str,
        output_path: Path,
        rate: str,
        pitch: str,
        volume: str,
        max_retries: int = 3,
    ) -> list:
        """Run edge-tts and extract word timestamps from WordBoundary events.

        Retries up to max_retries times on transient NoAudioReceived errors.
        """
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            communicate = edge_tts.Communicate(
                text,
                voice_id,
                rate=rate,
                pitch=pitch,
                volume=volume,
            )

            words = []

            try:
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
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = 2.0 * attempt
                    logger.warning(
                        "Edge-TTS attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt, max_retries, exc, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        "Edge-TTS failed after %d attempts: %s",
                        max_retries, exc,
                    )
        raise last_exc

    # --- Parallel chunking methods ---

    def _split_into_chunks(self, text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
        """Split text into sentence-level chunks for parallel synthesis.

        Handles Vietnamese punctuation. Never splits mid-word.
        Returns a single-element list if text is short or unsplittable.
        """
        if len(text) <= MIN_CHUNK_TEXT_LENGTH:
            return [text]

        sentences = _SENTENCE_SPLIT_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return [text]

        # Merge sentences into chunks respecting max_chars
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) + 1 > max_chars and current_parts:
                chunks.append(" ".join(current_parts))
                current_parts = [sent]
                current_len = len(sent)
            else:
                current_parts.append(sent)
                current_len += len(sent) + 1

        if current_parts:
            chunks.append(" ".join(current_parts))

        # If we ended up with just 1 chunk, no benefit from parallelism
        if len(chunks) <= 1:
            return [text]

        return chunks

    async def _generate_chunks(
        self,
        chunks: list[str],
        voice_id: str,
        rate: str,
        pitch: str,
        volume: str,
    ) -> list[dict]:
        """Synthesize multiple text chunks in parallel via asyncio.gather().

        Each chunk is synthesized independently with per-chunk caching.
        Returns list of {audio_path, words, duration, chunk_index} in order.
        Uses both a local semaphore and the global threading semaphore.
        """
        sem = asyncio.Semaphore(CHUNK_SEMAPHORE_LIMIT)
        chunk_dir = CACHE_DIR / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        async def _synth_one(idx: int, chunk_text: str) -> dict:
            # Per-chunk cache check
            ck = self._cache_key(chunk_text, voice_id, rate, pitch, volume)
            cached_audio = CACHE_DIR / f"{ck}.mp3"
            cached_meta = CACHE_DIR / f"{ck}.json"

            if cached_audio.exists() and cached_meta.exists():
                logger.debug("Chunk %d cache hit: %s", idx, ck)
                meta = json.loads(cached_meta.read_text())
                return {
                    "audio_path": str(cached_audio),
                    "words": meta.get("raw_words") or meta.get("words", []),
                    "duration": float(meta.get("duration", 0.0)),
                    "chunk_index": idx,
                    "cache_hit": True,
                }

            # Synthesize with semaphore control
            chunk_path = chunk_dir / f"{ck}_c{idx}.mp3"
            async with sem:
                # Acquire global semaphore (blocking in thread context)
                _EDGE_TTS_GLOBAL_SEMAPHORE.acquire()
                try:
                    words = await self._generate(
                        text=chunk_text,
                        voice_id=voice_id,
                        output_path=chunk_path,
                        rate=rate,
                        pitch=pitch,
                        volume=volume,
                    )
                finally:
                    _EDGE_TTS_GLOBAL_SEMAPHORE.release()

            # Read duration from generated audio
            info = sf.info(str(chunk_path))
            duration = info.duration

            if not words:
                words = self._proportional_spread(chunk_text.split(), 0.0, duration)
            words = self._coerce_words(words)

            # Write per-chunk cache
            cache_data = {
                "duration": round(duration, 3),
                "raw_words": words,
                "voice": voice_id,
                "rate": rate,
                "pitch": pitch,
                "volume": volume,
            }
            shutil.copy2(chunk_path, cached_audio)
            cached_meta.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2))

            return {
                "audio_path": str(chunk_path),
                "words": words,
                "duration": duration,
                "chunk_index": idx,
                "cache_hit": False,
            }

        tasks = [_synth_one(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        return list(results)

    def _merge_chunk_audio(self, chunk_results: list[dict], output_path: Path) -> float:
        """Concatenate chunk audio files into a single gapless file via PCM concat.

        Decodes each MP3 chunk to PCM numpy arrays, concatenates them,
        and writes the result. Returns total duration in seconds.
        """
        all_pcm: list[np.ndarray] = []
        sample_rate = None

        for cr in chunk_results:
            audio_path = cr["audio_path"]
            data, sr = sf.read(audio_path, dtype="float32")
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                # Resample if needed (unlikely — same voice/config)
                logger.warning(
                    "Sample rate mismatch: expected %d, got %d for chunk %d. Skipping resample.",
                    sample_rate, sr, cr["chunk_index"],
                )
            all_pcm.append(data)

        if not all_pcm or sample_rate is None:
            raise RuntimeError("No audio data from chunks")

        merged = np.concatenate(all_pcm)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), merged, sample_rate)

        total_duration = len(merged) / sample_rate
        return total_duration

    def _merge_chunk_words(self, chunk_results: list[dict]) -> list[dict]:
        """Merge word timestamps from all chunks into a continuous timeline.

        Each chunk's words are offset by the cumulative PCM duration of prior chunks.
        """
        merged: list[dict] = []
        cumulative_offset = 0.0

        for cr in sorted(chunk_results, key=lambda x: x["chunk_index"]):
            for w in cr["words"]:
                merged.append({
                    "word": w["word"],
                    "start": round(float(w["start"]) + cumulative_offset, 3),
                    "end": round(float(w["end"]) + cumulative_offset, 3),
                })
            cumulative_offset += cr["duration"]

        return merged

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

    def _postprocess_word_timestamps(self, words: list, duration: float) -> tuple[list, dict, dict, dict]:
        """Normalize and drift-correct word timestamps before subtitle rendering."""
        raw_words = self._coerce_words(words)
        raw_stats = self._timing_stats(raw_words, duration)

        normalized_words, normalize_info = self._normalize_word_timestamps(raw_words)
        normalized_stats = self._timing_stats(normalized_words, duration)

        corrected_words, drift_info = self._correct_end_drift(normalized_words, duration)
        final_words = self._enforce_monotonic(corrected_words)
        final_stats = self._timing_stats(final_words, duration)

        correction = {
            "normalize": normalize_info,
            "drift_correction": drift_info,
            "drift_before_sec": round(float(normalized_stats.get("tail_drift_sec") or 0.0), 3),
            "drift_after_sec": round(float(final_stats.get("tail_drift_sec") or 0.0), 3),
            "changed_word_count": _count_word_timing_changes(raw_words, final_words),
        }
        return final_words, correction, raw_stats, final_stats

    def _coerce_words(self, words: list) -> list[dict]:
        """Parse raw word entries into a minimal canonical structure."""
        normalized = []
        for item in words:
            token = str(item.get("word", "") or "").strip()
            if not token:
                continue
            try:
                start = float(item.get("start", 0.0) or 0.0)
                end = float(item.get("end", start) or start)
            except (TypeError, ValueError):
                continue

            if start < 0.0:
                start = 0.0
            if end < start:
                end = start
            normalized.append({"word": token, "start": start, "end": end})
        return normalized

    def _normalize_word_timestamps(self, words: list[dict]) -> tuple[list[dict], dict]:
        """
        Enforce monotonic boundaries, clamp minimum word duration, and smooth
        micro/large gaps before drift correction.
        """
        normalized = []
        overlap_fixes = 0
        micro_gap_fixes = 0
        large_gap_fixes = 0
        clamped_duration = 0
        shift_applied_sec = 0.0

        prev_end = 0.0
        for item in words:
            start = float(item.get("start", 0.0) or 0.0)
            end = float(item.get("end", start) or start)
            if normalized:
                if start < prev_end:
                    start = prev_end
                    overlap_fixes += 1

                gap = start - prev_end
                if 0.0 < gap < TIMING_MICRO_GAP_SEC:
                    start = prev_end
                    micro_gap_fixes += 1
                elif gap > TIMING_MAX_GAP_SEC:
                    squeeze = gap - TIMING_MAX_GAP_SEC
                    start -= squeeze
                    end -= squeeze
                    shift_applied_sec += squeeze
                    large_gap_fixes += 1

            min_end = start + TIMING_MIN_WORD_DURATION_SEC
            if end < min_end:
                end = min_end
                clamped_duration += 1

            normalized.append(
                {
                    "word": item["word"],
                    "start": round(start, 3),
                    "end": round(end, 3),
                }
            )
            prev_end = normalized[-1]["end"]

        summary = {
            "overlap_fixes": overlap_fixes,
            "micro_gap_fixes": micro_gap_fixes,
            "large_gap_fixes": large_gap_fixes,
            "min_duration_clamps": clamped_duration,
            "total_shift_applied_sec": round(shift_applied_sec, 3),
        }
        return normalized, summary

    def _correct_end_drift(self, words: list[dict], duration: float) -> tuple[list[dict], dict]:
        """
        Apply guarded timeline scaling and optional tail ramp shift when
        last word end drifts too far from the real audio duration.
        """
        if not words or duration <= 0.0:
            return words, {"applied": False, "reason": "no_words_or_duration"}

        first_start = float(words[0]["start"])
        last_end = float(words[-1]["end"])
        drift_before = float(duration) - last_end
        target_end = max(first_start + TIMING_MIN_WORD_DURATION_SEC, float(duration) - TIMING_TAIL_MARGIN_SEC)

        if abs(drift_before) < TIMING_DRIFT_THRESHOLD_SEC:
            return words, {
                "applied": False,
                "reason": "below_threshold",
                "drift_before_sec": round(drift_before, 3),
                "drift_after_sec": round(drift_before, 3),
            }

        span = max(last_end - first_start, TIMING_EPS)
        raw_scale = (target_end - first_start) / span
        scale = min(max(raw_scale, TIMING_SCALE_MIN), TIMING_SCALE_MAX)

        scaled = []
        for item in words:
            start = first_start + (float(item["start"]) - first_start) * scale
            end = first_start + (float(item["end"]) - first_start) * scale
            scaled.append({"word": item["word"], "start": start, "end": end})

        corrected = self._enforce_monotonic(scaled)
        ramp_shift = 0.0
        for _ in range(3):
            residual = target_end - float(corrected[-1]["end"])
            if abs(residual) < 0.02:
                break

            remaining_budget = max(TIMING_TOTAL_RAMP_SHIFT_LIMIT_SEC - abs(ramp_shift), 0.0)
            if remaining_budget <= TIMING_EPS:
                break
            step_cap = min(TIMING_MAX_RAMP_SHIFT_SEC, remaining_budget)
            step = min(max(residual, -step_cap), step_cap)
            corrected = self._apply_tail_ramp_shift(corrected, step)
            corrected = self._enforce_monotonic(corrected)
            ramp_shift += step

        drift_after = float(duration) - float(corrected[-1]["end"])
        return corrected, {
            "applied": True,
            "drift_before_sec": round(drift_before, 3),
            "drift_after_sec": round(drift_after, 3),
            "target_end_sec": round(target_end, 3),
            "raw_scale": round(raw_scale, 4),
            "applied_scale": round(scale, 4),
            "ramp_shift_sec": round(ramp_shift, 3),
        }

    def _apply_tail_ramp_shift(self, words: list[dict], shift_sec: float) -> list[dict]:
        """Shift timeline progressively so early words move less than late words."""
        if not words or abs(shift_sec) <= TIMING_EPS:
            return words

        first_start = float(words[0]["start"])
        span = max(float(words[-1]["end"]) - first_start, TIMING_EPS)
        shifted = []
        for item in words:
            start = float(item["start"])
            end = float(item["end"])
            progress_start = min(max((start - first_start) / span, 0.0), 1.0)
            progress_end = min(max((end - first_start) / span, 0.0), 1.0)
            shifted.append(
                {
                    "word": item["word"],
                    "start": start + (shift_sec * progress_start),
                    "end": end + (shift_sec * progress_end),
                }
            )
        return shifted

    def _enforce_monotonic(self, words: list[dict]) -> list[dict]:
        """Final guardrail: non-negative times, monotonic starts, and min durations."""
        fixed = []
        prev_end = 0.0
        for item in words:
            token = str(item.get("word", "") or "").strip()
            if not token:
                continue

            try:
                start = max(0.0, float(item.get("start", 0.0) or 0.0))
                end = float(item.get("end", start) or start)
            except (TypeError, ValueError):
                continue

            if fixed and start < prev_end:
                start = prev_end
            end = max(end, start + TIMING_MIN_WORD_DURATION_SEC)

            rounded_start = round(start, 3)
            rounded_end = round(max(end, rounded_start + TIMING_MIN_WORD_DURATION_SEC), 3)
            fixed.append({"word": token, "start": rounded_start, "end": rounded_end})
            prev_end = rounded_end
        return fixed

    def _timing_stats(self, words: list, duration: float) -> dict:
        """Build timing diagnostics to audit subtitle alignment quality."""
        if not words:
            return {
                "word_count": 0,
                "audio_duration_sec": round(float(duration), 3),
                "first_start_sec": 0.0,
                "last_end_sec": 0.0,
                "tail_drift_sec": round(float(duration), 3),
                "mean_word_dur_sec": 0.0,
                "p95_word_dur_sec": 0.0,
                "max_gap_sec": 0.0,
                "gap_gt_120ms": 0,
            }

        starts = [float(item.get("start", 0.0) or 0.0) for item in words]
        ends = [float(item.get("end", 0.0) or 0.0) for item in words]
        durations = [max(0.0, e - s) for s, e in zip(starts, ends)]

        gaps = []
        for idx in range(1, len(words)):
            gap = max(0.0, starts[idx] - ends[idx - 1])
            gaps.append(gap)

        p95 = _percentile(durations, 95.0)
        mean = sum(durations) / len(durations) if durations else 0.0
        first_start = min(starts) if starts else 0.0
        last_end = max(ends) if ends else 0.0
        tail_drift = float(duration) - last_end

        return {
            "word_count": len(words),
            "audio_duration_sec": round(float(duration), 3),
            "first_start_sec": round(first_start, 3),
            "last_end_sec": round(last_end, 3),
            "tail_drift_sec": round(tail_drift, 3),
            "mean_word_dur_sec": round(mean, 3),
            "p95_word_dur_sec": round(p95, 3),
            "max_gap_sec": round(max(gaps) if gaps else 0.0, 3),
            "gap_gt_120ms": sum(1 for gap in gaps if gap > 0.12),
        }

    def _log_timing_stats(self, cache_key: str, stats: dict, cache_hit: bool):
        logger.info(
            "Subtitle timing stats (%s, cache=%s): words=%d drift=%.3fs max_gap=%.3fs p95_word=%.3fs",
            cache_key,
            "hit" if cache_hit else "miss",
            int(stats.get("word_count") or 0),
            float(stats.get("tail_drift_sec") or 0.0),
            float(stats.get("max_gap_sec") or 0.0),
            float(stats.get("p95_word_dur_sec") or 0.0),
        )

    def _log_timing_correction(self, cache_key: str, correction: dict, cache_hit: bool):
        changed_words = int(correction.get("changed_word_count") or 0)
        drift_mode = correction.get("drift_correction") or {}
        normalize_mode = correction.get("normalize") or {}
        if changed_words <= 0 and not drift_mode.get("applied"):
            return
        logger.info(
            (
                "Subtitle timing correction (%s, cache=%s): changed=%d "
                "drift_before=%.3fs drift_after=%.3fs "
                "scale=%.4f ramp=%.3fs overlap_fix=%d gap_fix=%d"
            ),
            cache_key,
            "hit" if cache_hit else "miss",
            changed_words,
            float(correction.get("drift_before_sec") or 0.0),
            float(correction.get("drift_after_sec") or 0.0),
            float(drift_mode.get("applied_scale") or 1.0),
            float(drift_mode.get("ramp_shift_sec") or 0.0),
            int(normalize_mode.get("overlap_fixes") or 0),
            int(normalize_mode.get("large_gap_fixes") or 0),
        )

    def _maybe_dump_subtitle_debug(
        self,
        *,
        cache_key: str,
        text: str,
        output_path: Path,
        words: list,
        timing_stats: dict,
        voice: str,
        rate: str,
        pitch: str,
        volume: str,
        duration: float,
        cache_hit: bool,
        raw_words: Optional[list] = None,
        raw_timing_stats: Optional[dict] = None,
        timing_correction: Optional[dict] = None,
        alignment_mode_requested: Optional[str] = None,
        alignment_mode_effective: Optional[str] = None,
        alignment_mode_fallback_reason: Optional[str] = None,
    ):
        if not self.subtitle_debug_enabled:
            return

        SUBTITLE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        debug_file = SUBTITLE_DEBUG_DIR / f"{output_path.stem}_{cache_key}_{stamp}.json"
        payload = {
            "created_unix_ms": stamp,
            "cache_key": cache_key,
            "cache_hit": cache_hit,
            "audio_path": str(output_path),
            "voice": voice,
            "rate": rate,
            "pitch": pitch,
            "volume": volume,
            "duration_sec": round(float(duration), 3),
            "text": text,
            "raw_timing_stats": raw_timing_stats or {},
            "timing_stats": timing_stats,
            "timing_correction": timing_correction or {},
            "raw_words": raw_words or [],
            "words": words,
            "alignment_mode_requested": alignment_mode_requested,
            "alignment_mode_effective": alignment_mode_effective,
            "alignment_mode_fallback_reason": alignment_mode_fallback_reason,
        }
        debug_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        logger.debug("Subtitle debug dump written: %s", debug_file)


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _count_word_timing_changes(before: list[dict], after: list[dict], tolerance: float = 0.001) -> int:
    if not before and not after:
        return 0
    if len(before) != len(after):
        return max(len(before), len(after))

    changed = 0
    for left, right in zip(before, after):
        left_word = str(left.get("word", "") or "")
        right_word = str(right.get("word", "") or "")
        if left_word != right_word:
            changed += 1
            continue

        left_start = float(left.get("start", 0.0) or 0.0)
        right_start = float(right.get("start", 0.0) or 0.0)
        left_end = float(left.get("end", 0.0) or 0.0)
        right_end = float(right.get("end", 0.0) or 0.0)
        if abs(left_start - right_start) > tolerance or abs(left_end - right_end) > tolerance:
            changed += 1
    return changed


def default_voice_for_language(language: Optional[str]) -> str:
    raw = str(language or "").strip().lower()
    if not raw:
        return DEFAULT_VOICE
    primary = raw.split("-", 1)[0].split("_", 1)[0]
    return DEFAULT_VOICE_BY_LANGUAGE.get(primary, DEFAULT_VOICE)
