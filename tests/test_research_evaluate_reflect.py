"""
Standalone tests for the evaluate+reflect loop in run_research() (STAGE 4).
Mocks chat_completion so no network calls are made.

Run with: .venv/bin/python tests/test_research_evaluate_reflect.py
"""

import sys
import os
import json
import types
import importlib

# ---------------------------------------------------------------------------
# Bootstrap: make the project importable without installing it
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pass(name: str) -> None:
    print(f"PASS  {name}")

def _fail(name: str, reason: str) -> None:
    print(f"FAIL  {name}: {reason}")

# ---------------------------------------------------------------------------
# Lazy import so each test can patch before the module does anything real
# ---------------------------------------------------------------------------

def _load_research_module():
    """Import (or reload) research_agent after sys.modules surgery."""
    import src.agent.research_agent as m
    importlib.reload(m)
    return m

# ---------------------------------------------------------------------------
# Test 1 — _phase_evaluate: sufficient facts → returns []
# ---------------------------------------------------------------------------

def test_evaluate_sufficient_facts():
    name = "_phase_evaluate: fact_count >= target returns []"
    from unittest.mock import patch

    # fact_count == target, but even if the LLM were called the mock says sufficient
    sufficient_json = json.dumps({"sufficient": True, "missing_angles": []})

    with patch("src.llm_client.chat_completion", return_value=sufficient_json) as mock_cc:
        from src.agent.research_agent import _phase_evaluate

        result = _phase_evaluate("Test Topic", fact_count=12, target=12)

    try:
        # Even though fact_count == target, _phase_evaluate itself doesn't short-circuit
        # on count — the loop in run_research() does. The function still calls the LLM.
        # The contract being tested: when LLM says sufficient=true, returns [].
        assert result == [], f"expected [], got {result!r}"
        _pass(name)
    except AssertionError as exc:
        _fail(name, str(exc))


def test_evaluate_sufficient_flag_prevents_angles():
    name = "_phase_evaluate: sufficient=true with listed angles still returns []"
    from unittest.mock import patch

    # Even if the LLM oddly lists angles, sufficient=True must win
    resp = json.dumps({"sufficient": True, "missing_angles": ["origin story", "controversy"]})
    with patch("src.llm_client.chat_completion", return_value=resp):
        from src.agent.research_agent import _phase_evaluate
        result = _phase_evaluate("Test Topic", fact_count=10, target=12)

    try:
        assert result == [], f"expected [], got {result!r}"
        _pass(name)
    except AssertionError as exc:
        _fail(name, str(exc))


# ---------------------------------------------------------------------------
# Test 2 — _phase_evaluate: malformed JSON → returns [] (no crash)
# ---------------------------------------------------------------------------

def test_evaluate_malformed_json():
    name = "_phase_evaluate: malformed JSON from LLM returns []"
    from unittest.mock import patch

    with patch("src.llm_client.chat_completion", return_value="not json"):
        from src.agent.research_agent import _phase_evaluate
        try:
            result = _phase_evaluate("Test Topic", fact_count=5, target=12)
            assert result == [], f"expected [], got {result!r}"
            _pass(name)
        except Exception as exc:
            _fail(name, f"raised {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Test 3 — _phase_evaluate: LLM raises exception → returns [] (no crash)
# ---------------------------------------------------------------------------

def test_evaluate_llm_exception():
    name = "_phase_evaluate: LLM raises Exception returns []"
    from unittest.mock import patch

    with patch("src.llm_client.chat_completion", side_effect=Exception("timeout")):
        from src.agent.research_agent import _phase_evaluate
        try:
            result = _phase_evaluate("Test Topic", fact_count=5, target=12)
            assert result == [], f"expected [], got {result!r}"
            _pass(name)
        except Exception as exc:
            _fail(name, f"raised {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Test 4 — Dedup logic: 5 facts + extra 2 new + 1 duplicate → 7, not 8
# ---------------------------------------------------------------------------

def test_dedup_logic():
    name = "Dedup logic: 5 facts + (2 new + 1 duplicate) extra_facts → 7 total"

    def _make_fact(text: str) -> dict:
        return {
            "fact_text": text,
            "source": "web",
            "score": 0.7,
            "hook_text": text[:50],
            "source_url": "http://example.com",
            "reason_tags": ["extracted"],
        }

    facts = [_make_fact(f"Unique fact number {i} with enough text to pass length check") for i in range(5)]

    extra_facts = [
        _make_fact("Unique fact number 0 with enough text to pass length check"),  # DUPLICATE of facts[0]
        _make_fact("Brand new fact A that is not in the original list at all ok"),
        _make_fact("Brand new fact B that is not in the original list at all ok"),
    ]

    # Replicate the dedup logic verbatim from run_research() STAGE 4
    existing_prefixes = {f["fact_text"][:60].lower() for f in facts}
    for f in extra_facts:
        if f["fact_text"][:60].lower() not in existing_prefixes:
            facts.append(f)
            existing_prefixes.add(f["fact_text"][:60].lower())

    try:
        assert len(facts) == 7, f"expected 7 facts, got {len(facts)}"
        _pass(name)
    except AssertionError as exc:
        _fail(name, str(exc))


# ---------------------------------------------------------------------------
# Test 5 — Loop cap: _MAX_RESEARCH_ITERATIONS=1 → evaluate called at most once
# ---------------------------------------------------------------------------

def test_loop_cap():
    """
    Simulate the Stage 8 gap-fill loop with max_research_iterations=1.
    Patch _phase_evaluate to return missing angles every time (would loop forever
    without the cap), and _phase_reflect_crawl to return 1 new page each call.
    Count how many times _phase_evaluate is called — must be exactly 1.
    """
    name = "Loop cap: max_research_iterations=1 → evaluate called at most once"
    from unittest.mock import patch, MagicMock
    from src.agent.contracts import GroundedFact

    evaluate_call_count = 0

    def fake_evaluate(topic, fact_count, target):
        nonlocal evaluate_call_count
        evaluate_call_count += 1
        return ["origin story", "controversy"]

    def fake_reflect_crawl(topic, missing, language, existing_urls, emit, extraction_hint=""):
        return []  # Return empty — loop should still break after max iterations

    fake_cfg = {
        "max_research_iterations": 1,
        "min_grounded_facts": 12,
        "grounding_fuzzy_threshold": 0.80,
        "max_final_facts": 12,
        "chunk_size_chars": 3000,
        "chunk_min_chars": 100,
        "chunk_overlap_chars": 200,
    }

    # Build a minimal GroundedFact list below min_grounded_facts so the loop runs
    facts = [
        GroundedFact(
            fact_id=f"f{i}",
            claim=f"Fact {i} with enough text to pass dedup length check ok",
            verbatim_evidence=f"Evidence {i} from a source page with enough text here.",
            source_url=f"https://example.com/page{i}",
            source_domain="example.com",
            authority_tier=1,
            extraction_confidence=0.8,
            final_score=0.8,
            topic_match=True,
            grounded=True,
        )
        for i in range(3)  # only 3 facts — below the 12 minimum, loop should proceed
    ]

    with patch("src.agent.research_agent._phase_evaluate", side_effect=fake_evaluate), \
         patch("src.agent.research_agent._phase_reflect_crawl", side_effect=fake_reflect_crawl):

        # Replicate the Stage 8 loop as written in run_research()
        import src.agent.research_agent as ra

        final = list(facts)
        fact_ok = len(final) >= fake_cfg["min_grounded_facts"]
        iteration = 1
        while (not fact_ok) and iteration < fake_cfg["max_research_iterations"]:
            missing = ra._phase_evaluate("topic", len(final), fake_cfg["min_grounded_facts"])
            if not missing:
                break
            extra_raw = ra._phase_reflect_crawl("topic", missing, "en-US", set(), None)
            if not extra_raw:
                break
            iteration += 1

    try:
        assert evaluate_call_count == 1, (
            f"expected evaluate called exactly 1 time, got {evaluate_call_count}"
        )
        _pass(name)
    except AssertionError as exc:
        _fail(name, str(exc))


# ---------------------------------------------------------------------------
# Test 6 — _phase_evaluate with must_cover returns missing angles as subset
# ---------------------------------------------------------------------------

def test_evaluate_surfaces_must_cover_angles():
    """
    _phase_evaluate with must_cover angles should return missing angles that
    are a subset of the required must_cover angles.
    """
    name = "_phase_evaluate returns missing_angles as subset of must_cover"
    from unittest.mock import patch
    import json

    # Simulate: LLM says angles are missing
    missing_response = json.dumps({
        "sufficient": False,
        "missing_angles": ["cursed technique", "Sukuna possession"]
    })

    with patch("src.llm_client.chat_completion", return_value=missing_response):
        from src.agent.research_agent import _phase_evaluate
        result = _phase_evaluate(
            "Megumi Fushiguro",
            fact_count=2,
            target=12,
            user_prompt="Explain Megumi's cursed technique, comparing with Sukuna",
            must_cover=["cursed technique", "Sukuna possession"],
        )

    try:
        assert "cursed technique" in result, f"expected 'cursed technique' in result: {result}"
        assert "Sukuna possession" in result, f"expected 'Sukuna possession' in result: {result}"
        _pass(name)
    except AssertionError as exc:
        _fail(name, str(exc))


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Research agent evaluate+reflect loop — edge case tests")
    print("=" * 60)

    test_evaluate_sufficient_facts()
    test_evaluate_sufficient_flag_prevents_angles()
    test_evaluate_malformed_json()
    test_evaluate_llm_exception()
    test_dedup_logic()
    test_loop_cap()
    test_evaluate_surfaces_must_cover_angles()

    print("=" * 60)
    print("Done.")
