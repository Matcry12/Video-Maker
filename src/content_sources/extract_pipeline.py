"""Local extraction pipeline for compact source units.

This module deliberately does the heavy narrowing work before any LLM call:
- normalize source drafts into documents
- split documents into compact sentence clusters
- score locally for short-video interestingness
- dedupe near-duplicates lexically
- select top candidates for model ranking
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from .interest_ranker import RankCandidate
from .models import SourceDocument, SourceUnit

SPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z0-9À-ỹ]+", flags=re.UNICODE)
CONTRAST_MARKERS = {
    "but",
    "however",
    "yet",
    "despite",
    "although",
    "still",
    "while",
    "nhưng",
    "tuy",
    "dù",
}
EXTREME_MARKERS = {
    "largest",
    "smallest",
    "first",
    "last",
    "deadliest",
    "oldest",
    "fastest",
    "biggest",
    "strongest",
    "largest",
    "cao nhất",
    "lớn nhất",
    "đầu tiên",
    "cuối cùng",
}
HIGH_SIGNAL_TERMS = {
    "war",
    "killed",
    "collapsed",
    "destroyed",
    "empire",
    "prison",
    "explosion",
    "assassinated",
    "murder",
    "disaster",
    "legend",
    "mystery",
    "king",
    "queen",
    "battle",
    "death",
    "sụp đổ",
    "chiến tranh",
    "bí ẩn",
    "thảm họa",
}


def build_source_documents_from_draft(source_draft: dict[str, Any]) -> list[SourceDocument]:
    source = str(source_draft.get("source") or "source_1").strip() or "source_1"
    language = str(source_draft.get("language") or "en-US").strip() or "en-US"
    sections = source_draft.get("sections") or []
    documents: list[SourceDocument] = []

    for index, section in enumerate(sections, start=1):
        if not isinstance(section, dict):
            continue
        text = _clean_text(section.get("text"))
        if not text:
            continue
        doc_id = str(section.get("section_id") or f"d{index:03d}").strip() or f"d{index:03d}"
        title = _clean_text(section.get("title")) or f"Section {index}"
        documents.append(
            SourceDocument(
                doc_id=doc_id,
                title=title,
                url=str(section.get("source_url") or source_draft.get("source_url") or "").strip(),
                text=text,
                language=language,
                position=index,
                meta={
                    "source": source,
                    "rank": section.get("rank", index),
                    "token_estimate": section.get("token_estimate"),
                },
            )
        )
    return documents


def extract_source_units_from_draft(
    source_draft: dict[str, Any],
    *,
    max_chars_per_unit: int = 420,
    min_chars_per_unit: int = 120,
    dedupe_threshold: float = 0.88,
    keep_top_k: int = 40,
) -> dict[str, Any]:
    source = str(source_draft.get("source") or "source_1").strip() or "source_1"
    topic_query = _clean_text(source_draft.get("topic_query"))
    language = str(source_draft.get("language") or "en-US").strip() or "en-US"
    documents = build_source_documents_from_draft(source_draft)

    units: list[SourceUnit] = []
    for document in documents:
        units.extend(
            _segment_document_to_units(
                document,
                source=source,
                topic_query=topic_query,
                language=language,
                max_chars_per_unit=max_chars_per_unit,
                min_chars_per_unit=min_chars_per_unit,
            )
        )

    scored_units = [_score_source_unit_local(unit) for unit in units]
    deduped_units, removed_count = dedupe_source_units(scored_units, threshold=dedupe_threshold)
    top_units = select_top_source_units(deduped_units, keep_top_k=keep_top_k)

    return {
        "documents": [_model_dump(item) for item in documents],
        "units": [_model_dump(item) for item in deduped_units],
        "top_units": [_model_dump(item) for item in top_units],
        "meta": {
            "document_count": len(documents),
            "unit_count": len(units),
            "deduped_unit_count": len(deduped_units),
            "removed_duplicates": removed_count,
            "selected_top_units": len(top_units),
            "dedupe_threshold": threshold_to_str(dedupe_threshold),
        },
    }


def select_top_source_units(units: list[SourceUnit], *, keep_top_k: int = 40) -> list[SourceUnit]:
    ordered = sorted(
        units,
        key=lambda item: (
            -float(item.local_score or 0.0),
            item.position,
            item.unit_id,
        ),
    )
    return ordered[: max(1, int(keep_top_k))]


def dedupe_source_units(units: list[SourceUnit], *, threshold: float = 0.88) -> tuple[list[SourceUnit], int]:
    kept: list[SourceUnit] = []
    removed = 0
    kept_exact: set[str] = set()
    kept_tokens: list[set[str]] = []

    ordered = sorted(units, key=lambda item: (-float(item.local_score or 0.0), item.unit_id))
    for unit in ordered:
        exact_key = _exact_key(unit.text)
        if not exact_key:
            removed += 1
            continue
        if exact_key in kept_exact:
            removed += 1
            continue
        token_set = set(_tokenize(unit.text))
        near_dup = False
        for ref in kept_tokens:
            if _jaccard_similarity(token_set, ref) >= threshold:
                near_dup = True
                break
        if near_dup:
            removed += 1
            continue
        kept.append(unit)
        kept_exact.add(exact_key)
        kept_tokens.append(token_set)
    return kept, removed


def to_rank_candidates(units: list[SourceUnit | dict[str, Any]]) -> list[RankCandidate]:
    results: list[RankCandidate] = []
    for raw in units or []:
        unit = raw if isinstance(raw, SourceUnit) else SourceUnit(**raw)
        results.append(
            RankCandidate(
                candidate_id=unit.unit_id,
                text=unit.text,
                title=unit.title,
                url=unit.url,
                source=unit.source,
                topic_query=unit.topic_query,
                language=unit.language,
                local_score=unit.local_score,
                local_signals=dict(unit.local_signals or {}),
            )
        )
    return results


def _segment_document_to_units(
    document: SourceDocument,
    *,
    source: str,
    topic_query: str,
    language: str,
    max_chars_per_unit: int,
    min_chars_per_unit: int,
) -> list[SourceUnit]:
    sentences = _split_sentences(document.text)
    if not sentences:
        return []

    units: list[SourceUnit] = []
    buffer: list[str] = []
    position = 0

    def flush_buffer():
        nonlocal buffer, position
        text = _clean_text(" ".join(buffer))
        buffer = []
        if not text:
            return
        position += 1
        units.append(
            SourceUnit(
                unit_id=f"{document.doc_id}_u{position:03d}",
                source=source,
                topic_query=topic_query,
                doc_id=document.doc_id,
                title=document.title,
                url=document.url,
                text=text,
                language=language,
                position=position,
            )
        )

    for sentence in sentences:
        candidate = _clean_text(sentence)
        if not candidate:
            continue
        joined = _clean_text(" ".join(buffer + [candidate]))
        if buffer and len(joined) > max_chars_per_unit:
            flush_buffer()
            joined = candidate
        buffer.append(candidate)
        if len(joined) >= min_chars_per_unit:
            flush_buffer()

    if buffer:
        flush_buffer()

    return units


def _score_source_unit_local(unit: SourceUnit) -> SourceUnit:
    text = unit.text
    tokens = _tokenize(text)
    token_count = len(tokens)
    lowered = text.casefold()
    has_number = bool(re.search(r"\b\d[\d,./-]*\b", text))
    has_year = bool(re.search(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b", text))
    contrast_hits = sum(1 for marker in CONTRAST_MARKERS if marker in lowered)
    extreme_hits = sum(1 for marker in EXTREME_MARKERS if marker in lowered)
    signal_hits = sum(1 for marker in HIGH_SIGNAL_TERMS if marker in lowered)
    quote_hits = text.count('"') + text.count("'") + text.count("“") + text.count("”")
    uppercase_tokens = sum(1 for token in tokens if len(token) > 2 and token[:1].isupper())
    long_word_ratio = (
        sum(1 for token in tokens if len(token) >= 8) / max(token_count, 1)
        if token_count else 0.0
    )

    brevity_score = 0.9 if 12 <= token_count <= 45 else 0.72 if 8 <= token_count <= 60 else 0.5
    number_score = 0.12 if has_number else 0.0
    year_score = 0.08 if has_year else 0.0
    contrast_score = min(contrast_hits * 0.08, 0.16)
    extreme_score = min(extreme_hits * 0.09, 0.18)
    signal_score = min(signal_hits * 0.06, 0.18)
    quote_score = 0.05 if quote_hits else 0.0
    entity_score = min(uppercase_tokens * 0.02, 0.10)
    rarity_score = min(long_word_ratio * 0.25, 0.10)

    score = (
        brevity_score * 0.35
        + number_score
        + year_score
        + contrast_score
        + extreme_score
        + signal_score
        + quote_score
        + entity_score
        + rarity_score
    )
    unit.local_signals = {
        "token_count": token_count,
        "has_number": has_number,
        "has_year": has_year,
        "contrast_hits": contrast_hits,
        "extreme_hits": extreme_hits,
        "signal_hits": signal_hits,
        "quote_hits": quote_hits,
        "uppercase_tokens": uppercase_tokens,
        "long_word_ratio": round(long_word_ratio, 4),
        "brevity_score": round(brevity_score, 4),
    }
    unit.local_score = round(max(0.0, min(score, 1.0)), 4)
    return unit


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\n+", " ", str(text or "")).strip()
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(normalized) if part.strip()]


def _clean_text(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "")).strip()


def _tokenize(text: str) -> list[str]:
    return WORD_RE.findall(_clean_text(text).lower())


def _exact_key(text: str) -> str:
    return " ".join(_tokenize(text))


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / float(len(left | right))


def threshold_to_str(value: float) -> str:
    return f"{float(value):.2f}"


def _model_dump(value: BaseModel | SourceDocument | SourceUnit) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()
