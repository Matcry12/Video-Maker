"""Split a single-block narration into ~5-second phrase windows with per-window image keywords."""
from __future__ import annotations

import re

_STOPWORD_PROPERS = frozenset({
    # Conjunctive adverbs / subordinators
    "Unlike", "Because", "Despite", "However", "Although", "Though",
    "Moreover", "Furthermore", "Meanwhile", "Otherwise", "Therefore",
    "Instead", "Thus", "Hence", "Whereas", "While", "After", "Before",
    "During", "Since", "Until", "When", "Where", "Which",
    # Common sentence starters that are not proper nouns
    "But", "And", "The", "This", "That", "These", "Those",
    "His", "Her", "Its", "Our", "Their", "Your", "With",
    "Not", "Even", "Just", "Also", "Then", "From", "Into",
})

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
_PROPER_NOUN = re.compile(r'\b[A-Z][a-z]{2,}\b')


def _window_keywords(text: str, topic: str) -> list[str]:
    """Extract proper-noun keywords from a window, filtered and topic-prefixed."""
    propers = [w for w in _PROPER_NOUN.findall(text) if w not in _STOPWORD_PROPERS]
    seen: set[str] = set()
    keywords: list[str] = []
    for p in propers[:4]:
        kw = f"{topic} {p}".strip()
        if kw not in seen:
            seen.add(kw)
            keywords.append(kw)
    if not keywords:
        keywords.append(topic)
    return keywords


def _chop_by_words(text: str, max_words: int) -> list[str]:
    """Split text into chunks of at most max_words words at comma/semicolon or word boundary."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        # Try to break at a comma boundary within the chunk
        chunk_words = words[start:end]
        break_at = -1
        for i in range(len(chunk_words) - 1, max(len(chunk_words) // 2 - 1, -1), -1):
            if chunk_words[i].endswith(",") or chunk_words[i].endswith(";"):
                break_at = i + 1
                break
        if break_at > 0:
            end = start + break_at
        chunks.append(" ".join(words[start:end]))
        start = end
    return chunks


def split_into_windows(
    block_text: str,
    topic: str,
    max_words_per_window: int = 40,
) -> list[dict]:
    """Split narration into ~5-second phrase windows (~40 words each).

    Handles TTS scripts that use commas instead of periods — falls back to
    word-count chopping when a sentence exceeds max_words_per_window.

    Returns list of dicts: {text, image_keywords, word_start, word_end}
    """
    if not block_text or not block_text.strip():
        return []

    # Split on sentence-ending punctuation first
    raw_sentences = _SENTENCE_SPLIT.split(block_text.strip())

    # Expand any over-long "sentence" by word-count chopping
    sentences: list[str] = []
    for s in raw_sentences:
        if len(s.split()) > max_words_per_window:
            sentences.extend(_chop_by_words(s, max_words_per_window))
        else:
            sentences.append(s)

    windows: list[dict] = []
    buf_sentences: list[str] = []
    buf_words: list[str] = []
    word_cursor = 0

    for sentence in sentences:
        s_words = sentence.split()
        if buf_words and len(buf_words) + len(s_words) > max_words_per_window:
            w_start = word_cursor - len(buf_words)
            windows.append({
                "text": " ".join(buf_sentences),
                "image_keywords": _window_keywords(" ".join(buf_sentences), topic),
                "word_start": w_start,
                "word_end": word_cursor,
            })
            buf_sentences = []
            buf_words = []
        buf_sentences.append(sentence)
        buf_words.extend(s_words)
        word_cursor += len(s_words)

    if buf_sentences:
        w_start = word_cursor - len(buf_words)
        windows.append({
            "text": " ".join(buf_sentences),
            "image_keywords": _window_keywords(" ".join(buf_sentences), topic),
            "word_start": w_start,
            "word_end": word_cursor,
        })

    return windows
