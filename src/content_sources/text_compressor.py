"""Text compression utilities for reducing LLM token consumption."""

import re

_URL_RE = re.compile(r"https?://\S+")
_CITATION_RE = re.compile(r"\[\d+\]")
_BOILERPLATE_RE = re.compile(
    r"^\s*(references|see also|external links|further reading|navigation menu|"
    r"retrieved from|jump to navigation|jump to search|edit source)\s*$",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def compress_for_llm(text: str, max_chars: int = 2000) -> str:
    """Strip boilerplate, URLs, citations, redundant whitespace.

    Deduplicates sentences using Jaccard similarity (threshold 0.85) and
    truncates at a sentence boundary within max_chars.
    """
    if not text:
        return ""

    cleaned = _URL_RE.sub("", text)
    cleaned = _CITATION_RE.sub("", cleaned)

    lines = cleaned.splitlines()
    filtered_lines: list[str] = []
    for line in lines:
        if _BOILERPLATE_RE.search(line):
            continue
        filtered_lines.append(line)
    cleaned = " ".join(filtered_lines)

    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    sentences = _split_sentences(cleaned)
    deduped = _deduplicate_sentences(sentences, threshold=0.85)
    compressed = " ".join(deduped)

    if len(compressed) <= max_chars:
        return compressed

    truncated = compressed[:max_chars]
    last_boundary = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )
    if last_boundary > max_chars // 2:
        truncated = truncated[: last_boundary + 1]
    return truncated.strip()


def estimate_tokens(text: str) -> int:
    """Quick token estimate: word_count * 1.35 (same as wikipedia_source.py)."""
    words = len(str(text or "").split())
    return max(1, int(round(words * 1.35)))


def _jaccard_similarity(a: set, b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries."""
    normalized = re.sub(r"\n+", " ", text or "").strip()
    parts = _SENTENCE_BOUNDARY_RE.split(normalized)
    return [p.strip() for p in parts if p and p.strip()]


def _deduplicate_sentences(sentences: list[str], threshold: float = 0.85) -> list[str]:
    """Remove sentences that are highly similar to an already-seen sentence."""
    seen_token_sets: list[set] = []
    result: list[str] = []
    for sentence in sentences:
        tokens = set(sentence.lower().split())
        duplicate = any(
            _jaccard_similarity(tokens, seen) >= threshold for seen in seen_token_sets
        )
        if not duplicate:
            result.append(sentence)
            seen_token_sets.append(tokens)
    return result
