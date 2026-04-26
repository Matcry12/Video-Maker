from src.agent.grounding import is_grounded, topic_mentioned
from src.agent.contracts import GroundedFact
from src.agent.research_agent import (
    _verify_grounding,
    _score_and_rank_facts,
    _jaccard,
    _check_quality_floor,
)

SOURCE = (
    "Subaru Natsuki is transported to a fantasy world. His ability, Return by Death, "
    "lets him return to a save point upon dying. The cost is enormous psychological trauma."
)


def test_grounded_exact():
    assert is_grounded("His ability, Return by Death, lets him return to a save point", SOURCE) is True


def test_grounded_normalized():
    # Punctuation / whitespace differences
    assert is_grounded('his ability — return by death — lets him return to a save point.', SOURCE) is True


def test_not_grounded_hallucinated():
    assert is_grounded("Al shares a soul with Subaru", SOURCE) is False


def test_too_short_rejected():
    assert is_grounded("fantasy", SOURCE) is False


def test_topic_mentioned_alias():
    assert topic_mentioned("Subaru's return by death ability", ["Re:Zero", "Subaru"]) is True


def test_topic_not_mentioned():
    assert topic_mentioned("A generic statement about souls.", ["Re:Zero"]) is False


# ---------------------------------------------------------------------------
# Plan 02 additions — research_agent integration
# ---------------------------------------------------------------------------


def test_verify_grounding_keeps_real_fact():
    raw = [{
        "claim": "Subaru Natsuki's ability, Return by Death, lets him return to a save point.",
        "verbatim_evidence": "His ability, Return by Death, lets him return to a save point upon dying.",
        "confidence": 0.9,
        "source_url": "https://en.wikipedia.org/wiki/Re:Zero",
        "source_domain": "en.wikipedia.org",
        "authority_tier": 1,
        "parent_text": (
            "Subaru Natsuki is transported to a fantasy world. His ability, Return by Death, "
            "lets him return to a save point upon dying."
        ),
    }]
    kept, stats = _verify_grounding(raw, ["Re:Zero", "Subaru"], fuzzy_threshold=0.80)
    assert len(kept) == 1
    assert kept[0].grounded is True
    assert stats["failed_grounding"] == 0


def test_verify_grounding_drops_hallucination():
    raw = [{
        "claim": "Al and Subaru share a soul.",
        "verbatim_evidence": "Al and Subaru share the same soul across dimensions.",
        "confidence": 0.85,
        "source_url": "https://mystiqora.com/rezero-paradox",
        "source_domain": "mystiqora.com",
        "authority_tier": 4,
        "parent_text": "Re:Zero is an anime series. It features Subaru Natsuki as the protagonist.",
    }]
    kept, stats = _verify_grounding(raw, ["Re:Zero", "Subaru"], fuzzy_threshold=0.80)
    assert len(kept) == 0
    assert stats["failed_grounding"] == 1


def test_score_and_rank_prefers_tier1():
    f1 = GroundedFact(
        fact_id="F001", claim="wikipedia fact", verbatim_evidence="x" * 50,
        source_url="https://en.wikipedia.org/wiki/X", source_domain="en.wikipedia.org",
        authority_tier=1, extraction_confidence=0.7, final_score=0.7 * 1.5,
        grounded=True, topic_match=True, reason_tags=["grounded"],
    )
    f2 = GroundedFact(
        fact_id="F002", claim="blog fact", verbatim_evidence="y" * 50,
        source_url="https://randomblog.com/x", source_domain="randomblog.com",
        authority_tier=4, extraction_confidence=0.9, final_score=0.9 * 0.25,
        grounded=True, topic_match=True, reason_tags=["grounded"],
    )
    ranked = _score_and_rank_facts([f2, f1], max_final=5)
    assert ranked[0].claim == "wikipedia fact"


def test_jaccard_dedup():
    assert _jaccard("Subaru uses Return by Death to reset", "Subaru uses Return by Death for resets") > 0.5
    assert _jaccard("completely different topic", "totally unrelated material") < 0.3


def test_quality_floor_passes():
    cfg = {"min_grounded_facts": 2, "min_tier_diversity": 2, "require_min_tier": 2}
    facts = [
        GroundedFact("F001", "a", "x" * 40, "u1", "wikipedia.org", 1, 0.9, 1.35, True, True, []),
        GroundedFact("F002", "b", "x" * 40, "u2", "screenrant.com", 2, 0.8, 0.8, True, True, []),
    ]
    warnings = []
    fact_ok, div_ok = _check_quality_floor(facts, cfg, warnings)
    assert fact_ok is True
    assert div_ok is True


def test_quality_floor_fails_on_tier():
    cfg = {"min_grounded_facts": 2, "min_tier_diversity": 2, "require_min_tier": 2}
    facts = [
        GroundedFact("F001", "a", "x" * 40, "u1", "randomblog.com", 4, 0.9, 0.225, True, True, []),
        GroundedFact("F002", "b", "x" * 40, "u2", "otherblog.com", 4, 0.9, 0.225, True, True, []),
    ]
    warnings = []
    fact_ok, _ = _check_quality_floor(facts, cfg, warnings)
    assert fact_ok is False
