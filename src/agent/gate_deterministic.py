"""Deterministic (rule-based) quality checks that run BEFORE the LLM judge.

Each check returns (score_0_to_10, issue_message_or_None). Total is 50 pts.
"""
from __future__ import annotations

import re

from src.agent.entity_sanitizer import forbidden_entities, is_contaminated


def check_grounding(script: dict) -> tuple[int, str | None]:
    """10 pts if citation rate >= 80%, 5 pts if >= 60%, 7 pts (neutral) if no citation map."""
    cmap = script.get("__citation_map__")
    if cmap is None:
        # Citations not available for this run — neutral score, not a penalty
        return 7, None
    rate = float(cmap.get("citation_rate") or 0.0)
    if rate >= 0.80:
        return 10, None
    if rate >= 0.60:
        return 5, f"Citation rate only {rate:.0%} (target 80%)"
    return 0, f"Citation rate {rate:.0%} — writer produced ungrounded content"


def check_contamination(script: dict, plan) -> tuple[int, str | None]:
    """10 pts if no foreign franchise names, 0 otherwise."""
    topic = getattr(plan, "topic", "") or ""
    aliases = list(getattr(plan, "entity_aliases", []) or [])
    fb = forbidden_entities(topic, topic_aliases=aliases)
    bad: list[str] = []
    for block in script.get("blocks") or []:
        text = block.get("text") or ""
        contam, hits = is_contaminated(text, fb)
        if contam:
            bad.extend(hits)
    if not bad:
        return 10, None
    uniq = sorted(set(bad))
    return 0, f"Cross-franchise contamination: {', '.join(uniq[:5])}"


def check_hook_specificity(script: dict) -> tuple[int, str | None]:
    """10 pts if first sentence contains a number, proper noun, or concrete
    detail. 0 if it opens with a generic phrase."""
    blocks = script.get("blocks") or []
    if not blocks:
        return 0, "No blocks"
    first_text = (blocks[0].get("text") or "").strip()
    if not first_text:
        return 0, "Empty first block"
    first_sent = re.split(r"(?<=[.!?])\s+", first_text, maxsplit=1)[0]
    generic_openers = [
        "in this video", "today we",  "let's talk about", "have you ever",
        "you won't believe", "welcome to", "in the world of", "it's no secret",
        "in the realm of", "every fan knows",
    ]
    lower = first_sent.lower()
    if any(g in lower for g in generic_openers):
        return 0, f"Hook opens with generic phrase: {first_sent[:80]!r}"
    has_digit = any(c.isdigit() for c in first_sent)
    has_proper = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", first_sent))
    if has_digit or has_proper:
        return 10, None
    return 5, f"Hook lacks a specific anchor (number or proper noun): {first_sent[:80]!r}"


def check_loop_back(script: dict) -> tuple[int, str | None]:
    """10 pts if last sentence/block connects back to first without copying it.
    0 pts if last sentence is a near-duplicate of the first (Jaccard > 0.5).
    For single-block scripts, compares first and last sentences within the block.
    """
    blocks = script.get("blocks") or []
    stopwords = {
        "the", "a", "an", "is", "are", "of", "in", "on", "at", "to", "and",
        "or", "with", "for", "from", "this", "that", "as", "was", "were", "by",
        "he", "she", "it", "they", "but",
    }
    def _tokens(s):
        return {w.lower() for w in re.findall(r"[A-Za-zÀ-ỹ]+", s) if w.lower() not in stopwords and len(w) > 3}
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    if len(blocks) == 1:
        # Single-block design: compare first vs last sentence within the block
        text = blocks[0].get("text", "")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) < 2:
            return 5, None
        first_tok = _tokens(sentences[0])
        last_tok = _tokens(sentences[-1])
        j = _jaccard(first_tok, last_tok)
        if j > 0.5:
            return 0, f"Duplicate ending: last sentence copies the hook (Jaccard {j:.2f})"
        overlap = first_tok & last_tok
        if len(overlap) >= 2:
            return 10, None
        if len(overlap) == 1:
            return 5, "Weak loop-back (1 shared keyword between first and last sentence)"
        return 3, "No thematic loop-back between first and last sentence"

    # Multi-block: compare first and last blocks
    first = _tokens(blocks[0].get("text", ""))
    last = _tokens(blocks[-1].get("text", ""))
    j = _jaccard(first, last)
    if j > 0.5:
        return 0, f"Duplicate ending: last block copies the first (Jaccard {j:.2f})"
    overlap = first & last
    if len(overlap) >= 2:
        return 10, None
    if len(overlap) == 1:
        return 5, "Weak loop-back (1 shared keyword)"
    return 0, "No loop-back between first and last block"


def check_sentence_length(script: dict) -> tuple[int, str | None]:
    """10 pts if average sentence <= 20 words, 5 pts <= 28, 0 otherwise."""
    sentences = []
    for b in script.get("blocks") or []:
        text = b.get("text") or ""
        for s in re.split(r"(?<=[.!?])\s+", text):
            s = s.strip()
            if s:
                sentences.append(s)
    if not sentences:
        return 0, "No sentences"
    lengths = [len(s.split()) for s in sentences]
    avg = sum(lengths) / len(lengths)
    if avg <= 20:
        return 10, None
    if avg <= 28:
        return 5, f"Sentences slightly long (avg {avg:.1f} words; target <= 20)"
    return 0, f"Sentences too long for voice-over (avg {avg:.1f} words)"


def check_must_cover(script: dict, plan) -> tuple[int, str | None]:
    """Reward scripts that visibly address every must_cover angle.

    10 pts if all angles appear as substring (normalized) in the script text.
    5 pts if >= 60% of angles are covered.
    0 pts otherwise (or if must_cover is empty — neutral, not penalized).
    """
    must_cover = list(getattr(plan, "must_cover", []) or [])
    if not must_cover:
        return 10, None  # no required angles → full score (neutral)

    script_text = " ".join(
        block.get("text", "") for block in (script.get("blocks") or [])
    ).lower()

    covered = 0
    for angle in must_cover:
        if angle.lower() in script_text:
            covered += 1

    ratio = covered / len(must_cover)
    if ratio >= 1.0:
        return 10, None
    if ratio >= 0.6:
        return 5, f"Only {covered}/{len(must_cover)} required angles covered in script"
    return 0, f"Missing required angles in script: {covered}/{len(must_cover)} covered"


def run_deterministic_checks(script: dict, plan) -> tuple[int, list[str]]:
    """Return (total_score_0_to_60, issue_messages)."""
    checks = [
        check_grounding(script),
        check_contamination(script, plan),
        check_hook_specificity(script),
        check_loop_back(script),
        check_sentence_length(script),
        check_must_cover(script, plan),  # NEW — max 10 pts, total max now 60
    ]
    total = sum(s for s, _ in checks)
    issues = [m for _, m in checks if m]
    return total, issues
