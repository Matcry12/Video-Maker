"""src.broll — B-roll Shorts pipeline for the Psychology Facts channel.

Converts a topic string into a finished mp4 with stock footage, narration,
animated karaoke captions, BGM, and YouTube metadata.

Public API:
    build_broll(topic, *, target_words, bgm_vol, skip_dupe_check, emit) -> BrollResult
    BrollResult   — dataclass returned by build_broll
    BrollDuplicateError — raised when topic is already in the content index
"""

from .builder import build_broll, BrollResult, BrollDuplicateError

__all__ = ["build_broll", "BrollResult", "BrollDuplicateError"]
