import pytest

from src.agent.gate_deterministic import (
    check_grounding, check_contamination, check_hook_specificity,
    check_loop_back, check_sentence_length, run_deterministic_checks,
    check_natural_speech,
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


def test_natural_speech_clean():
    """Hyped-fan tone with mixed lengths and a hype word → full score."""
    script = {"blocks": [{"text": (
        "Tusk Act 4 is literally broken. "
        "It cancels GER before it even fires. "
        "Watch the clip and you'll see Johnny laugh as the world freezes. "
        "Game over. "
        "Giorno never had a chance."
    )}]}
    score, issue = check_natural_speech(script)
    assert score == 10, f"expected 10, got {score} (issue: {issue})"
    assert issue is None


def test_natural_speech_forbidden_vocab():
    """Essay vocab triggers deduction."""
    script = {"blocks": [{"text": (
        "Tusk's infinity defies causality. Furthermore, GER's ability to nullify "
        "actions establishes a paradigm shift in stand combat. However, the "
        "outcome is therefore inevitable."
    )}]}
    score, issue = check_natural_speech(script)
    assert score < 10, f"expected penalty, got {score}"
    assert "essay vocab" in (issue or "")


def test_natural_speech_uniform_pacing():
    """All-similar sentence lengths → robotic pacing penalty."""
    script = {"blocks": [{"text": (
        "Act one fires nail bullets. "
        "Act two drills holes deeper. "
        "Act three opens wormholes wide. "
        "Act four cancels infinity. "
        "Johnny wins the duel."
    )}]}
    score, issue = check_natural_speech(script)
    assert score < 10, f"expected penalty for uniform pacing, got {score}"
    assert issue is not None


def test_natural_speech_no_hype_words():
    """Decent prose but missing hype beats → small deduction."""
    script = {"blocks": [{"text": (
        "Johnny rides into Diego's territory with purpose. "
        "Tusk is a stand he barely controls, yet it grows alongside his journey. "
        "Each act unlocks something more dangerous."
    )}]}
    score, _ = check_natural_speech(script)
    assert score == 8, f"expected 10 - 2 (no hype) = 8, got {score}"


def test_natural_speech_empty():
    script = {"blocks": [{"text": ""}]}
    score, issue = check_natural_speech(script)
    assert score == 0
    assert issue == "No text"
