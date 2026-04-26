"""Normalize image_keywords to keep topic and franchise context anchored."""

from __future__ import annotations

import re

_TOPIC_STOPWORDS = {
    "the", "and", "or", "for", "from", "with", "without", "into",
    "onto", "about", "over", "under", "after", "before", "during",
    "what", "when", "where", "which", "who", "whom", "whose",
    "this", "that", "these", "those", "top", "best", "worst",
    "guide", "explained", "simple", "simply", "facts", "fact",
    "to", "of", "in", "on", "at", "by",
}


def enrich_keywords(
    keywords: list[str],
    topic: str,
    topic_category: str = "",
) -> list[str]:
    """Safe keyword enrichment. Never returns None or an empty list
    when input had items — falls back to the original on any error."""
    if not keywords:
        return []
    try:
        return _enrich_keywords_core(keywords, topic, topic_category)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("enrich_keywords failed (%s); using originals", exc)
        return list(keywords)


def _enrich_keywords_core(
    keywords: list[str],
    topic: str,
    topic_category: str = "",
) -> list[str]:
    """Ensure every image keyword contains the main topic context."""
    del topic_category  # reserved for future category-specific enrichment

    primary_name = _primary_name(topic)
    topic_tokens = _meaningful_tokens(primary_name)
    franchise = _extract_franchise(topic)

    enriched: list[str] = []
    for kw in keywords:
        raw = str(kw).strip()
        if not raw:
            continue

        lower = raw.lower()
        has_topic = any(token in lower for token in topic_tokens)
        has_franchise = bool(franchise) and franchise.lower() in lower

        if has_topic and (not franchise or has_franchise):
            enriched.append(raw)
            continue

        parts: list[str] = []
        if not has_topic:
            parts.append(primary_name)
        parts.append(raw)
        if franchise and not has_franchise:
            parts.append(franchise)
        enriched.append(" ".join(parts))

    seen: set[str] = set()
    result: list[str] = []
    for kw in enriched:
        key = kw.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(kw)
    return result


def _meaningful_tokens(topic: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9']+", topic.lower())
    return [
        token for token in tokens
        if len(token) >= 3 and token not in _TOPIC_STOPWORDS
    ]


def _primary_name(topic: str) -> str:
    cleaned = re.sub(r"\s*\([^)]*\)", "", topic).strip()
    cleaned = re.sub(r"\bfrom\s+.+$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or topic.strip()


def _extract_franchise(topic: str) -> str:
    match = re.search(r"\(([^)]+)\)", topic)
    if match:
        return match.group(1).strip()

    from_match = re.search(r"\bfrom\s+(.+)$", topic, flags=re.IGNORECASE)
    if from_match:
        return from_match.group(1).strip(" .:-")

    return ""
