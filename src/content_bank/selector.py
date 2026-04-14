"""Selection logic for composing scripts from fact bank."""

from __future__ import annotations

import random
from collections import defaultdict

from .models import FactCard

SUPPORTED_SELECTION_MODES = {"top", "balanced", "random_weighted"}


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _as_script_text(fact: FactCard) -> str:
    base = str(fact.fact_text or "").strip()
    hook = str(fact.hook_text or "").strip()
    if not base:
        return ""

    if not hook:
        return base

    # Add hook when base is short enough, otherwise keep concise base line.
    if len(base.split()) <= 22:
        merged = f"{base} {hook}".strip()
        return " ".join(merged.split())
    return base


def build_script_from_facts(selected_facts: list[FactCard], language: str = "en-US") -> dict:
    blocks = []
    for fact in selected_facts:
        text = _as_script_text(fact)
        if not text:
            continue
        blocks.append(
            {
                "text": text,
                "fact_id": fact.fact_id,
                "source_topic_id": fact.topic_id,
                "source_topic": fact.topic_label,
                "source_url": fact.source_url,
            }
        )

    return {
        "language": str(language or "en-US").strip() or "en-US",
        "blocks": blocks,
    }


def select_facts(
    facts: list[FactCard],
    *,
    pick_topics_count: int = 3,
    pick_facts_count: int | None = None,
    selection_mode: str = "balanced",
    exclude_used: bool = True,
) -> tuple[list[FactCard], dict]:
    mode = str(selection_mode or "balanced").strip().lower()
    if mode not in SUPPORTED_SELECTION_MODES:
        mode = "balanced"

    available = list(facts)
    if exclude_used:
        available = [item for item in available if item.status == "unused"]

    available = sorted(available, key=lambda item: (-float(item.score or 0.0), item.fact_id))

    requested_topics = _clamp_int(pick_topics_count or 1, 1, 50)
    requested_facts = _clamp_int(pick_facts_count or requested_topics, 1, 100)

    if not available:
        return [], {
            "selection_mode": mode,
            "requested_topics": requested_topics,
            "requested_facts": requested_facts,
            "selected_topics": 0,
            "selected_facts": 0,
            "exclude_used": bool(exclude_used),
        }

    if mode == "top":
        selected = available[:requested_facts]
    elif mode == "random_weighted":
        selected = _select_random_weighted(available, requested_facts)
    else:
        selected = _select_balanced(available, requested_topics, requested_facts)

    topic_ids = {item.topic_id for item in selected}
    meta = {
        "selection_mode": mode,
        "requested_topics": requested_topics,
        "requested_facts": requested_facts,
        "selected_topics": len(topic_ids),
        "selected_facts": len(selected),
        "exclude_used": bool(exclude_used),
    }
    return selected, meta


def _select_balanced(available: list[FactCard], requested_topics: int, requested_facts: int) -> list[FactCard]:
    by_topic: dict[str, list[FactCard]] = defaultdict(list)
    for item in available:
        by_topic[item.topic_id].append(item)

    ranked_topics = sorted(
        by_topic.keys(),
        key=lambda topic_id: -float(by_topic[topic_id][0].score or 0.0),
    )
    chosen_topics = ranked_topics[: max(1, min(requested_topics, len(ranked_topics)))]

    selected: list[FactCard] = []

    # Pass 1: pick top fact from each selected topic.
    for topic_id in chosen_topics:
        if len(selected) >= requested_facts:
            break
        bucket = by_topic.get(topic_id) or []
        if bucket:
            selected.append(bucket[0])

    # Pass 2: round-robin the rest from selected topics.
    cursor = {topic_id: 1 for topic_id in chosen_topics}
    while len(selected) < requested_facts:
        progressed = False
        for topic_id in chosen_topics:
            bucket = by_topic.get(topic_id) or []
            idx = cursor[topic_id]
            if idx < len(bucket):
                selected.append(bucket[idx])
                cursor[topic_id] = idx + 1
                progressed = True
                if len(selected) >= requested_facts:
                    break
        if not progressed:
            break

    # Pass 3: fill from remaining topics if still short.
    if len(selected) < requested_facts:
        already = {item.fact_id for item in selected}
        for item in available:
            if item.fact_id in already:
                continue
            selected.append(item)
            if len(selected) >= requested_facts:
                break

    return selected


def _select_random_weighted(available: list[FactCard], requested_facts: int) -> list[FactCard]:
    pool = list(available)
    selected: list[FactCard] = []

    while pool and len(selected) < requested_facts:
        weights = [max(float(item.score or 0.0), 0.01) for item in pool]
        chosen = random.choices(pool, weights=weights, k=1)[0]
        selected.append(chosen)
        pool = [item for item in pool if item.fact_id != chosen.fact_id]

    return selected
