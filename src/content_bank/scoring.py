"""Fact scoring and deduplication utilities for content bank."""

from __future__ import annotations

import re
from typing import Iterable

from .models import FactCard, ScoreBreakdown

_WORD_RE = re.compile(r"[A-Za-z0-9À-ỹ]+", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    raw = str(text or "").strip().lower()
    raw = _SPACE_RE.sub(" ", raw)
    return raw


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(normalize_text(text))


def exact_dedupe_key(text: str) -> str:
    tokens = tokenize(text)
    return " ".join(tokens)


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / float(len(left_set | right_set))


def _clamp01(value: float) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _clarity_score_from_text(text: str) -> float:
    words = tokenize(text)
    count = len(words)
    if count == 0:
        return 0.0
    if 8 <= count <= 24:
        return 0.85
    if 5 <= count <= 32:
        return 0.7
    if 3 <= count <= 42:
        return 0.55
    return 0.4


def _hook_strength_from_text(hook_text: str) -> float:
    words = tokenize(hook_text)
    count = len(words)
    if count == 0:
        return 0.45
    if 6 <= count <= 20:
        return 0.82
    if 3 <= count <= 28:
        return 0.68
    return 0.5


def _novelty_from_text(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 0.0
    unique_ratio = len(set(words)) / max(len(words), 1)
    return max(0.4, min(0.9, unique_ratio))


def _relevance_from_topic(fact_text: str, topic_label: str) -> float:
    fact_tokens = tokenize(fact_text)
    topic_tokens = tokenize(topic_label)
    if not fact_tokens:
        return 0.0
    if not topic_tokens:
        return 0.62
    overlap = jaccard_similarity(fact_tokens, topic_tokens)
    return max(0.55, min(0.9, 0.58 + overlap * 0.5))


def _score_from_breakdown(breakdown: ScoreBreakdown) -> float:
    # Weighted blend tuned for short-video usefulness.
    score = (
        breakdown.hook_strength * 0.30
        + breakdown.relevance * 0.25
        + breakdown.novelty * 0.20
        + breakdown.clarity * 0.15
        + breakdown.confidence * 0.10
    )
    return round(max(0.0, min(1.0, score)), 4)


def apply_scores(facts: list[FactCard]) -> list[FactCard]:
    """Fill score_breakdown + score for each fact card in-place."""
    for fact in facts:
        current = fact.score_breakdown or ScoreBreakdown()

        novelty = _clamp01(current.novelty) or _novelty_from_text(fact.fact_text)
        relevance = _clamp01(current.relevance) or _relevance_from_topic(fact.fact_text, fact.topic_label)
        hook_strength = _clamp01(current.hook_strength) or _hook_strength_from_text(fact.hook_text)
        clarity = _clamp01(current.clarity) or _clarity_score_from_text(fact.fact_text)
        confidence = _clamp01(current.confidence) or 0.62

        breakdown = ScoreBreakdown(
            novelty=round(novelty, 4),
            relevance=round(relevance, 4),
            hook_strength=round(hook_strength, 4),
            clarity=round(clarity, 4),
            confidence=round(confidence, 4),
        )
        fact.score_breakdown = breakdown
        fact.score = _score_from_breakdown(breakdown)

    return facts


def dedupe_fact_cards(
    candidates: list[FactCard],
    existing: list[FactCard] | None = None,
    *,
    near_dup_threshold: float = 0.88,
) -> tuple[list[FactCard], int]:
    """Dedupe against existing + candidate set using exact and near-duplicate checks."""
    existing = existing or []
    removed = 0

    existing_exact = {exact_dedupe_key(item.fact_text) for item in existing if item.fact_text}
    existing_tokens = [set(tokenize(item.fact_text)) for item in existing if item.fact_text]

    # Keep stronger facts first when candidates collide.
    ordered = sorted(candidates, key=lambda item: (-float(item.score or 0.0), item.fact_id))

    kept: list[FactCard] = []
    kept_exact: set[str] = set()
    kept_tokens: list[set[str]] = []

    for fact in ordered:
        exact_key = exact_dedupe_key(fact.fact_text)
        if not exact_key:
            removed += 1
            continue

        if exact_key in existing_exact or exact_key in kept_exact:
            removed += 1
            continue

        token_set = set(tokenize(fact.fact_text))
        near_dup = False

        for ref_set in existing_tokens:
            if jaccard_similarity(token_set, ref_set) >= near_dup_threshold:
                near_dup = True
                break

        if not near_dup:
            for ref_set in kept_tokens:
                if jaccard_similarity(token_set, ref_set) >= near_dup_threshold:
                    near_dup = True
                    break

        if near_dup:
            removed += 1
            continue

        kept.append(fact)
        kept_exact.add(exact_key)
        kept_tokens.append(token_set)

    return kept, removed
