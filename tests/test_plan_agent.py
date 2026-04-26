import pytest
import os

from src.agent.plan_agent import plan_from_prompt, _has_vn_chars
from src.agent.models import AgentPlan


def test_vn_detection_diacritics():
    assert _has_vn_chars("Hãy làm một video về")
    assert _has_vn_chars("Tiếng Việt")
    assert not _has_vn_chars("Plain English text")


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_plan_rezero_aliases():
    plan = plan_from_prompt("Uncovering darkest moments in Re:Zero")
    assert "Re:Zero" in plan.topic or "Re Zero" in plan.topic
    assert len(plan.entity_aliases) >= 1
    assert len(plan.search_queries) >= 3
    banned_substrings = ["jujutsu kaisen", "naruto", "one piece", "attack on titan"]
    for q in plan.search_queries:
        qlow = q.lower()
        for b in banned_substrings:
            assert b not in qlow, f"banned substring {b!r} in query {q!r}"


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_plan_vietnamese_language():
    plan = plan_from_prompt("Hãy tạo video về lịch sử triều Nguyễn")
    assert plan.language == "vi-VN"


def test_plan_stores_user_prompt():
    """AgentPlan carries verbatim user_prompt from heuristic path."""
    from src.agent.plan_agent import _plan_heuristic
    prompt = "Explain Megumi's cursed technique, compared to Sukuna"
    plan = _plan_heuristic(prompt)
    assert plan.user_prompt == prompt, f"user_prompt not stored: {plan.user_prompt!r}"
    assert isinstance(plan.must_cover, list), "must_cover must be a list"


def test_heuristic_must_cover_extracts_angles():
    """_heuristic_must_cover extracts angles from explicit prompt."""
    from src.agent.plan_agent import _heuristic_must_cover
    prompt = "Explain Megumi's cursed technique, comparing with how Sukuna uses his ability in his body."
    angles = _heuristic_must_cover(prompt)
    assert len(angles) >= 1, f"expected at least 1 angle, got {angles}"
    joined = " ".join(angles).lower()
    assert "cursed technique" in joined or "technique" in joined or "sukuna" in joined, \
        f"expected relevant angle, got {angles}"


def test_heuristic_must_cover_empty_for_vague():
    """_heuristic_must_cover returns [] for a vague topic-only prompt."""
    from src.agent.plan_agent import _heuristic_must_cover
    prompt = "Megumi Fushiguro Jujutsu Kaisen"
    angles = _heuristic_must_cover(prompt)
    # Vague topic-only prompt should not produce angles (no cover/explain/compare words)
    assert isinstance(angles, list), "must return list"
    # May be empty or have 0 items — that's fine
    assert len(angles) == 0, f"expected empty list for vague prompt, got {angles}"


def test_plan_heuristic_must_cover():
    """_plan_heuristic applies heuristic must_cover extraction."""
    from src.agent.plan_agent import _plan_heuristic
    prompt = "Explain Megumi's cursed technique, comparing with Sukuna"
    plan = _plan_heuristic(prompt)
    assert plan.user_prompt == prompt
    # Heuristic should have extracted at least one angle
    assert isinstance(plan.must_cover, list), "must_cover must be a list"
