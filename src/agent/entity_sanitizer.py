"""Foreign-entity contamination detection.

Given a topic like "Re:Zero" and text like
"Al and Subaru side by side JJK", this module detects that
"JJK" refers to a different franchise and should be dropped.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PATH = _PROJECT_ROOT / "data" / "known_entities.json"

_cached_registry: dict[str, list[str]] | None = None


def _load_registry(path: Path | None = None) -> dict[str, list[str]]:
    global _cached_registry
    if _cached_registry is not None:
        return _cached_registry
    try:
        p = path or _DEFAULT_PATH
        _cached_registry = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("entity_sanitizer: could not load %s: %s", path or _DEFAULT_PATH, exc)
        _cached_registry = {}
    return _cached_registry


def _normalize(s: str) -> str:
    # Lowercase and collapse punctuation to single spaces.
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _alias_tokens(alias: str) -> set[str]:
    return set(_normalize(alias).split())


def _text_matches_alias(text_norm: str, alias: str) -> bool:
    """Return True iff `alias` appears as a whole-token substring in `text_norm`.

    "naruto" matches "about naruto" but NOT "narutosaves".
    """
    alias_norm = _normalize(alias)
    if not alias_norm:
        return False
    return bool(re.search(rf"\b{re.escape(alias_norm)}\b", text_norm))


def forbidden_entities(topic: str, topic_aliases: list[str] | None = None,
                       categories: Iterable[str] = ("anime", "games")) -> list[str]:
    """Return all entity names from `known_entities.json` that are NOT
    the current topic. These are names to treat as contamination."""
    registry = _load_registry()
    self_tokens: set[str] = set()
    for alias in list(topic_aliases or []) + [topic]:
        self_tokens |= _alias_tokens(alias)

    forbidden: list[str] = []
    for cat in categories:
        for entity in registry.get(cat, []):
            if _alias_tokens(entity) & self_tokens:
                continue  # this entity is our topic
            forbidden.append(entity)
    return forbidden


def is_contaminated(text: str, forbidden: list[str]) -> tuple[bool, list[str]]:
    """Return (contaminated, hits). `hits` is the list of forbidden aliases
    that appeared in the text."""
    if not text or not forbidden:
        return False, []
    text_norm = _normalize(text)
    hits = [f for f in forbidden if _text_matches_alias(text_norm, f)]
    return (len(hits) > 0, hits)


def sanitize_list(items: list[str], forbidden: list[str]) -> tuple[list[str], list[str]]:
    """Drop items that contain forbidden entities. Return (kept, dropped)."""
    kept: list[str] = []
    dropped: list[str] = []
    for it in items:
        contam, _ = is_contaminated(it, forbidden)
        if contam:
            dropped.append(it)
        else:
            kept.append(it)
    return kept, dropped
