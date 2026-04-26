import pytest

from src.agent.gate_deterministic import (
    check_grounding, check_contamination, check_hook_specificity,
    check_loop_back, check_sentence_length, run_deterministic_checks,
    check_must_cover,
)
from src.agent.models import AgentPlan


def _plan(topic="Re:Zero", aliases=None):
    return AgentPlan(topic=topic, entity_aliases=aliases or ["Re Zero", "Rezero"])


def test_grounding_full_rate():
    script = {
        "blocks": [{"text": "A."}],
        "__citation_map__": {"citation_rate": 0.9},
    }
    score, _ = check_grounding(script)
    assert score == 10


def test_grounding_low_rate():
    script = {"__citation_map__": {"citation_rate": 0.3}}
    score, issue = check_grounding(script)
    assert score == 0
    assert "30%" in issue


def test_contamination_clean():
    script = {"blocks": [{"text": "Subaru uses Return by Death."}]}
    score, issue = check_contamination(script, _plan())
    assert score == 10
    assert issue is None


def test_contamination_detected():
    script = {"blocks": [{"text": "Al and Subaru look like JJK characters."}]}
    score, issue = check_contamination(script, _plan())
    assert score == 0
    assert "JJK" in issue


def test_hook_with_number():
    script = {"blocks": [{"text": "7 hidden details about Re:Zero will shock you."}]}
    score, _ = check_hook_specificity(script)
    assert score == 10


def test_hook_generic():
    script = {"blocks": [{"text": "In this video, we will discuss Re:Zero."}]}
    score, issue = check_hook_specificity(script)
    assert score == 0
    assert "generic" in issue.lower()


def test_loop_back_strong():
    script = {"blocks": [
        {"text": "Subaru's Return by Death ability has a hidden cost."},
        {"text": "Will Subaru ever truly escape Return by Death?"},
    ]}
    score, _ = check_loop_back(script)
    assert score == 10


def test_sentence_length_ok():
    script = {"blocks": [{"text": "Short sentence here. Another short one. Keep it brief."}]}
    score, _ = check_sentence_length(script)
    assert score == 10


def test_run_all_perfect():
    script = {
        "blocks": [
            {"text": "7 dark Re:Zero facts every fan missed."},
            {"text": "The first reveal comes from Tappei's own notes."},
            {"text": "Will you look at Re:Zero the same way again?"},
        ],
        "__citation_map__": {"citation_rate": 0.9},
    }
    total, issues = run_deterministic_checks(script, _plan())
    assert total >= 48


def _plan_with_must_cover(topic="Re:Zero", must_cover=None):
    plan = _plan(topic)
    plan.must_cover = must_cover or []
    return plan


def test_must_cover_all():
    """All must_cover angles present in script → 10 pts."""
    script = {"blocks": [
        {"text": "Subaru's Return by Death is a cursed technique. Sukuna possesses him."},
    ]}
    plan = _plan_with_must_cover(must_cover=["cursed technique", "Sukuna"])
    score, issue = check_must_cover(script, plan)
    assert score == 10, f"expected 10, got {score}"
    assert issue is None


def test_must_cover_partial():
    """Only 1 of 2 must_cover angles present → 5 pts (60% threshold)."""
    script = {"blocks": [
        {"text": "Subaru's Return by Death is a cursed technique and ability."},
    ]}
    plan = _plan_with_must_cover(must_cover=["cursed technique", "Sukuna possession"])
    score, issue = check_must_cover(script, plan)
    # 1/2 = 50% covered — below 60% threshold → 0 pts
    assert score == 0, f"expected 0 for 50% coverage, got {score}"


def test_must_cover_none():
    """Empty must_cover → 10 pts (neutral, not penalized)."""
    script = {"blocks": [{"text": "Some narration text here."}]}
    plan = _plan_with_must_cover(must_cover=[])
    score, issue = check_must_cover(script, plan)
    assert score == 10, f"expected 10 for empty must_cover, got {score}"
    assert issue is None
