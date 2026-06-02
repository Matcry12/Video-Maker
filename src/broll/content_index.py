"""Content index for the b-roll pipeline.

Tracks produced videos to prevent duplicate topics on the channel.
Index file: src/broll/content_index.json
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

_INDEX_PATH = Path(__file__).parent / "content_index.json"


def _normalize(topic: str) -> str:
    """Lowercase, strip all non-alphanumerics."""
    return re.sub(r"[^a-z0-9]", "", topic.lower())


def _load() -> dict:
    if _INDEX_PATH.exists():
        return json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    return {"videos": []}


def _save(data: dict) -> None:
    _INDEX_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def is_duplicate(topic: str) -> bool:
    """Return True if a video for this topic has already been produced."""
    norm = _normalize(topic)
    data = _load()
    for entry in data.get("videos", []):
        if entry.get("normalized_topic") == norm:
            return True
    return False


def append_entry(
    *,
    topic: str,
    niche: str,
    hook: str,
    output_file: str,
    bgm: str,
    youtube: dict | None = None,
    status: str = "draft",
) -> None:
    """Append a new entry to the content index and write it back."""
    data = _load()
    entry: dict = {
        "topic": topic,
        "normalized_topic": _normalize(topic),
        "niche": niche,
        "hook": hook,
        "output_file": output_file,
        "bgm": bgm,
        "date": datetime.date.today().isoformat(),
        "status": status,
    }
    if youtube is not None:
        entry["youtube"] = youtube
    data.setdefault("videos", []).append(entry)
    _save(data)
