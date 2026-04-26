"""Fact-grounding verification.

A grounded fact is one where the claimed `verbatim_evidence` substring
genuinely appears in the source text. We allow small deviations because
LLMs often drop periods, normalize quotes, or collapse whitespace.
"""
from __future__ import annotations

import difflib
import re


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\"\u201c\u201d\u2018\u2019']", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def is_grounded(evidence: str, source_text: str, fuzzy_threshold: float = 0.80, tier: int | None = None) -> bool:
    """Return True iff `evidence` appears in `source_text`.

    Strategy:
      1. Normalize both (lowercase, strip quotes, collapse whitespace).
      2. Exact substring match after normalization → True.
      3. If evidence is very short (< 30 chars), require exact match.
      4. Else, slide a window of len(evidence_norm) across the source and
         take the max SequenceMatcher ratio; pass if ≥ fuzzy_threshold.

    `tier` — when provided and tier >= 2, the threshold is relaxed to 0.65
    so fandom/narrative-prose sources aren't unfairly penalised vs Wikipedia.
    """
    if tier is not None and tier >= 2:
        fuzzy_threshold = min(fuzzy_threshold, 0.65)
    if not evidence or not source_text:
        return False
    ev = _normalize(evidence)
    src = _normalize(source_text)
    if len(ev) < 20:
        return False  # too short to be meaningful evidence
    if ev in src:
        return True
    if len(ev) < 30:
        return False

    # Fuzzy sliding window — only necessary when exact match fails
    # Step by 20 chars to keep cost bounded.
    best = 0.0
    step = max(20, len(ev) // 4)
    for start in range(0, max(1, len(src) - len(ev) + 1), step):
        window = src[start:start + len(ev) + 20]
        ratio = difflib.SequenceMatcher(None, ev, window).ratio()
        if ratio > best:
            best = ratio
        if best >= fuzzy_threshold:
            return True
    return best >= fuzzy_threshold


def topic_mentioned(claim: str, topic_aliases: list[str]) -> bool:
    """Return True iff `claim` mentions at least one topic alias."""
    if not claim:
        return False
    claim_norm = _normalize(claim)
    for alias in topic_aliases or []:
        a = _normalize(alias)
        if a and re.search(rf"\b{re.escape(a)}\b", claim_norm):
            return True
    return False
