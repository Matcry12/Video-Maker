import os
import pytest

from src.agent.trend_agent import _validate_brainstorm, _fallback_brainstorm


def test_validate_accepts_good_brainstorm():
    good = {
        "video_angle": "Uncovering the darkest moments in Re:Zero's history",
        "search_queries": ["Re:Zero darkest moments", "Subaru trauma", "Re:Zero psychological"],
        "skill_id": "dark_secrets",
        "hook_idea": "...",
    }
    assert _validate_brainstorm(good, "Re:Zero") is True


def test_validate_rejects_empty():
    assert _validate_brainstorm({}, "Naruto") is False


def test_validate_rejects_short_angle():
    bad = {"video_angle": "short", "search_queries": ["a", "b", "c"], "skill_id": "x"}
    assert _validate_brainstorm(bad, "Naruto") is False


def test_validate_rejects_no_title_reference():
    bad = {
        "video_angle": "Dark secrets about some anime you haven't heard of",
        "search_queries": ["generic q", "another q", "third q"],
        "skill_id": "dark_secrets",
    }
    assert _validate_brainstorm(bad, "Naruto") is False


def test_fallback_has_required_shape():
    fb = _fallback_brainstorm("Naruto")
    assert _validate_brainstorm(fb, "Naruto") is True
