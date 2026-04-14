"""
Whisper-based forced alignment for subtitle timing.

Uses faster-whisper to transcribe the audio and extract word-level timestamps,
then aligns the transcribed words back to the original text for precise timing.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "base"
_model_cache: dict[str, object] = {}


def _get_model(model_size: str):
    """Load and cache a faster-whisper model."""
    if model_size in _model_cache:
        return _model_cache[model_size]

    from faster_whisper import WhisperModel

    logger.info("Loading Whisper model '%s' (first use may download ~150MB)...", model_size)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    _model_cache[model_size] = model
    logger.info("Whisper model '%s' loaded.", model_size)
    return model


def _language_code(language: Optional[str]) -> Optional[str]:
    """Convert locale like 'en-US' or 'vi-VN' to ISO 639-1 code for whisper."""
    if not language:
        return None
    primary = language.strip().split("-")[0].split("_")[0].lower()
    return primary if primary else None


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy word comparison."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _align_to_original(
    whisper_words: list[dict],
    original_text: str,
) -> list[dict]:
    """
    Map whisper word timestamps back onto original text words.

    Whisper may produce slightly different tokens than the original text
    (e.g. "6,000" split into "6" + ",000", missing punctuation, merged words).
    This function aligns them so the subtitle text matches the original while
    timing comes from whisper.
    """
    orig_tokens = original_text.split()
    if not orig_tokens:
        return whisper_words

    if not whisper_words:
        return []

    orig_norm = [_normalize(t) for t in orig_tokens]
    whis_norm = [_normalize(w["word"]) for w in whisper_words]

    aligned = []
    wi = 0  # whisper index

    for oi, (orig_token, orig_n) in enumerate(zip(orig_tokens, orig_norm)):
        if not orig_n:
            # Pure punctuation token — attach to previous timing
            if aligned:
                aligned.append({
                    "word": orig_token,
                    "start": aligned[-1]["end"],
                    "end": round(aligned[-1]["end"] + 0.05, 3),
                })
            continue

        if wi >= len(whisper_words):
            # Past all whisper words: extend from last
            prev_end = aligned[-1]["end"] if aligned else 0.0
            aligned.append({
                "word": orig_token,
                "start": round(prev_end, 3),
                "end": round(prev_end + 0.15, 3),
            })
            continue

        # Try direct match
        if whis_norm[wi] == orig_n or (
            orig_n and whis_norm[wi] and (orig_n in whis_norm[wi] or whis_norm[wi] in orig_n)
        ):
            aligned.append({
                "word": orig_token,
                "start": whisper_words[wi]["start"],
                "end": whisper_words[wi]["end"],
            })
            wi += 1
            continue

        # Try merging consecutive whisper tokens to match one original token
        # e.g. whisper: "6" + ",000" → "6000" matches original "6000"
        merged_norm = whis_norm[wi]
        merge_end = wi
        found_merge = False
        for lookahead in range(wi + 1, min(wi + 5, len(whisper_words))):
            merged_norm += whis_norm[lookahead]
            merge_end = lookahead
            if merged_norm == orig_n or (
                orig_n and merged_norm and (orig_n in merged_norm or merged_norm in orig_n)
            ):
                # Merged match: use start of first token, end of last
                aligned.append({
                    "word": orig_token,
                    "start": whisper_words[wi]["start"],
                    "end": whisper_words[merge_end]["end"],
                })
                wi = merge_end + 1
                found_merge = True
                break
        if found_merge:
            continue

        # Try skipping one whisper token (whisper inserted an extra word)
        if wi + 1 < len(whisper_words) and (
            whis_norm[wi + 1] == orig_n
            or (orig_n and whis_norm[wi + 1] and (orig_n in whis_norm[wi + 1] or whis_norm[wi + 1] in orig_n))
        ):
            wi += 1  # skip the extra whisper token
            aligned.append({
                "word": orig_token,
                "start": whisper_words[wi]["start"],
                "end": whisper_words[wi]["end"],
            })
            wi += 1
            continue

        # No match found: interpolate between neighbors
        if aligned:
            prev_end = aligned[-1]["end"]
            next_start = whisper_words[wi]["start"] if wi < len(whisper_words) else prev_end + 0.3
            mid = (prev_end + next_start) / 2
            aligned.append({
                "word": orig_token,
                "start": round(prev_end, 3),
                "end": round(max(mid, prev_end + 0.05), 3),
            })
        else:
            aligned.append({
                "word": orig_token,
                "start": round(max(0.0, whisper_words[wi]["start"] - 0.15), 3),
                "end": round(whisper_words[wi]["start"], 3),
            })

    return aligned


def align_audio(
    audio_path: Path,
    original_text: str,
    *,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
) -> list[dict]:
    """
    Run whisper on audio to get word-level timestamps.

    Returns list of {"word": str, "start": float, "end": float}
    matching the format expected by editor.py's generate_ass().

    Returns empty list on failure so the caller can fall back.
    """
    if model_size is None:
        model_size = os.getenv("VM_WHISPER_MODEL", DEFAULT_MODEL_SIZE)

    try:
        model = _get_model(model_size)
    except Exception:
        logger.exception("Failed to load Whisper model '%s'; falling back.", model_size)
        return []

    language_code = _language_code(language)

    try:
        transcribe_kwargs = {"word_timestamps": True}
        if language_code:
            transcribe_kwargs["language"] = language_code

        segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)

        whisper_words = []
        for segment in segments:
            if not segment.words:
                continue
            for word_info in segment.words:
                whisper_words.append({
                    "word": word_info.word.strip(),
                    "start": round(word_info.start, 3),
                    "end": round(word_info.end, 3),
                })

        if not whisper_words:
            logger.warning("Whisper returned no word-level timestamps for '%s'.", audio_path)
            return []

        logger.info(
            "Whisper extracted %d words from '%s' (language=%s).",
            len(whisper_words),
            audio_path.name,
            info.language,
        )

        aligned = _align_to_original(whisper_words, original_text)
        return aligned

    except Exception:
        logger.exception("Whisper transcription failed for '%s'; falling back.", audio_path)
        return []
