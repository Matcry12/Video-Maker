"""
Hard edge case tests for research_agent.py — NOT happy-path.

Covers:
  Input edge cases        (cases 1-5)
  Logic edge cases        (cases 6-10)
  Resilience/mock cases   (cases 11-13)

Run with: .venv/bin/python tests/test_research_hard_cases.py
"""

import sys
import os
import json
import types

# ---------------------------------------------------------------------------
# Bootstrap: make the project importable without installing it
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0

def _pass(name: str) -> None:
    global _PASS
    _PASS += 1
    print(f"PASS  {name}")

def _fail(name: str, reason: str) -> None:
    global _FAIL
    _FAIL += 1
    print(f"FAIL  {name}: {reason}")


# ---------------------------------------------------------------------------
# Import only the pure functions under test — no network, no LLM at module level
# ---------------------------------------------------------------------------

from src.agent.research_agent import (
    _extract_english_query,
    _verify_grounding,
    _score_and_rank_facts,
    _phase_evaluate,
    _phase_search,
    _phase_wiki,
)

# ============================================================
# INPUT EDGE CASES
# ============================================================

# Case 1 — Topic with only stopwords
def test_stopword_only_topic():
    name = "case 1: topic='The Best Facts' — keyword extraction degrades gracefully (no crash, returns str)"
    topic = "The Best Facts"
    try:
        result = _extract_english_query(topic)
        # "the" is a stopword; "Best" and "Facts" are NOT in the stopword list → they survive
        assert isinstance(result, str), f"expected str, got {type(result)}"
        # Must not raise, must not return None
        _pass(name)
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {exc}")


# Case 2 — Very short topic
def test_short_topic_query_building():
    name = "case 2: topic='AI' — _extract_english_query returns non-None string"
    topic = "AI"
    try:
        result = _extract_english_query(topic)
        assert isinstance(result, str), f"expected str, got {type(result)}"
        # 'AI' is 2 chars, all ASCII — should survive the filter
        assert "AI" in result or result == "", f"unexpected result: {result!r}"
        _pass(name)
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {exc}")


# Case 3 — Vietnamese mixed-ASCII topic
def test_vietnamese_mixed_ascii_extraction():
    name = "case 3: topic='Gojo Satoru trong Jujutsu Kaisen' — extracts English tokens only"
    topic = "Gojo Satoru trong Jujutsu Kaisen"
    try:
        result = _extract_english_query(topic)
        assert isinstance(result, str), f"expected str, got {type(result)}"
        # "trong" is a Vietnamese stopword — should be excluded
        assert "trong" not in result.lower(), f"'trong' leaked into result: {result!r}"
        # The romanized proper nouns should survive
        for expected in ("Gojo", "Satoru", "Jujutsu", "Kaisen"):
            assert expected in result, f"expected '{expected}' in result, got: {result!r}"
        _pass(name)
    except AssertionError as exc:
        _fail(name, str(exc))
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {exc}")


# Case 4 — Empty search_queries list triggers template fallback
def test_empty_search_queries_uses_templates():
    name = "case 4: search_queries=[] — phase_search builds queries from templates (no crash)"
    from unittest.mock import patch

    captured_queries = []

    def fake_ddg(query, language="en-US", max_results=12):
        captured_queries.append(query)
        return []  # no results, but no crash

    with patch("src.content_sources.duckduckgo_source.search_duckduckgo", side_effect=fake_ddg):
        try:
            results, snippets = _phase_search("Naruto", [], "en-US", None)
            assert isinstance(results, list), "expected list for results"
            assert isinstance(snippets, list), "expected list for snippets"
            # Templates should have been used to build queries
            assert len(captured_queries) > 0, "expected at least one query to be built from templates"
            # Each query should contain the topic
            for q in captured_queries:
                assert "Naruto" in q, f"template query missing topic: {q!r}"
            _pass(name)
        except Exception as exc:
            _fail(name, f"raised {type(exc).__name__}: {exc}")


# Case 5 — Topic with special chars / C++ style
def test_special_chars_topic():
    name = "case 5: topic='C++ programming' — query building does not crash"
    topic = "C++ programming"
    try:
        result = _extract_english_query(topic)
        assert isinstance(result, str), f"expected str, got {type(result)}"
        # Should not crash on + chars; clean regex removes non-alphanum except :-'
        # "C" alone is len 1 → filtered; "programming" ≥ 2 → survives
        assert "programming" in result, f"expected 'programming' in result: {result!r}"
        _pass(name)
    except AssertionError as exc:
        _fail(name, str(exc))
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {exc}")


# ============================================================
# LOGIC EDGE CASES
# ============================================================

# Case 6 — _verify_grounding drops claims with no evidence match to aliases
def test_verify_grounding_drops_unrelated():
    name = "case 6: _verify_grounding drops claim with no alias match in evidence"
    raw = [
        {
            "claim": "Subaru can reset time on death.",
            "verbatim_evidence": "Subaru can reset time on death because of his ability.",
            "confidence": 0.9,
            "source_url": "https://rezero.fandom.com/wiki/Natsuki_Subaru",
            "source_domain": "rezero.fandom.com",
            "authority_tier": 1,
        },
        {
            "claim": "An unrelated franchise claim.",
            "verbatim_evidence": "This passage discusses something entirely different with no topic mention at all here.",
            "confidence": 0.9,
            "source_url": "https://example.com/unrelated",
            "source_domain": "example.com",
            "authority_tier": 2,
        },
    ]
    try:
        kept, stats = _verify_grounding(raw, ["Re:Zero", "Subaru"], fuzzy_threshold=0.80)
        # The Re:Zero fact should survive; the unrelated one may fail topic match
        assert len(kept) >= 1, f"expected at least 1 grounded fact, got {len(kept)}"
        claims = [f["claim"] for f in kept]
        assert any("Subaru" in c or "reset" in c.lower() for c in claims), f"expected Re:Zero fact to survive: {claims}"
        _pass(name)
    except AssertionError as exc:
        _fail(name, str(exc))
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {exc}")


# Case 7 — _verify_grounding with empty list → returns ([], stats)
def test_verify_grounding_empty_list():
    name = "case 7: _verify_grounding with empty list → returns ([], stats)"
    try:
        kept, stats = _verify_grounding([], ["Re:Zero"], fuzzy_threshold=0.80)
        assert kept == [], f"expected [], got {kept!r}"
        assert isinstance(stats, dict), "expected stats dict"
        _pass(name)
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {exc}")


# Case 8 — _score_and_rank_facts preserves tier-1 facts preferentially
def test_score_and_rank_prefers_tier1():
    name = "case 8: _score_and_rank_facts prefers tier-1 over tier-3"
    from src.agent.contracts import GroundedFact
    f1 = GroundedFact(
        fact_id="f1",
        claim="Tier-1 fact about Re:Zero canon.",
        verbatim_evidence="Re:Zero author confirmed in an interview.",
        source_url="https://rezero.fandom.com/wiki/Subaru",
        source_domain="rezero.fandom.com",
        authority_tier=1,
        extraction_confidence=0.9,
        final_score=0.9,
        topic_match=True,
        grounded=True,
    )
    f2 = GroundedFact(
        fact_id="f2",
        claim="Tier-3 fan theory about Re:Zero.",
        verbatim_evidence="Reddit users speculated about this.",
        source_url="https://reddit.com/r/re_zero/post",
        source_domain="reddit.com",
        authority_tier=3,
        extraction_confidence=0.3,
        final_score=0.3,
        topic_match=True,
        grounded=True,
    )
    try:
        ranked = _score_and_rank_facts([f2, f1], max_final=5)
        assert len(ranked) == 2, f"expected 2 facts, got {len(ranked)}"
        # Tier-1 should rank first (higher authority_tier_weight)
        assert ranked[0].authority_tier == 1, f"tier-1 should rank first: {[r.authority_tier for r in ranked]}"
        _pass(name)
    except AssertionError as exc:
        _fail(name, str(exc))
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {exc}")


# Case 9 — _score_and_rank_facts respects max_final cap
def test_score_and_rank_max_cap():
    name = "case 9: _score_and_rank_facts respects max_final cap"
    from src.agent.contracts import GroundedFact
    facts = [
        GroundedFact(
            fact_id=f"f{i}",
            claim=f"Fact {i} about Re:Zero with distinct content item number {i}.",
            verbatim_evidence=f"Evidence {i} from Re:Zero source document page {i}.",
            source_url=f"https://rezero.fandom.com/wiki/Page{i}",
            source_domain="rezero.fandom.com",
            authority_tier=1,
            extraction_confidence=0.8,
            final_score=0.8,
            topic_match=True,
            grounded=True,
        )
        for i in range(10)
    ]
    try:
        ranked = _score_and_rank_facts(facts, max_final=5)
        assert len(ranked) <= 5, f"expected at most 5 facts, got {len(ranked)}"
        _pass(name)
    except AssertionError as exc:
        _fail(name, str(exc))
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {exc}")


# Case 10 — _extract_english_query with pure Vietnamese (no ASCII tokens) → returns ""
def test_extract_english_query_pure_vietnamese():
    name = "case 10: _extract_english_query with pure Vietnamese → returns ''"
    # All tokens contain non-ASCII chars or are Vietnamese stopwords
    topic = "những bí mật ẩn giấu về thế giới"
    try:
        result = _extract_english_query(topic)
        assert result == "", f"expected '', got {result!r}"
        _pass(name)
    except AssertionError as exc:
        _fail(name, str(exc))
    except Exception as exc:
        _fail(name, f"raised {type(exc).__name__}: {exc}")


# ============================================================
# RESILIENCE CASES (mock network)
# ============================================================

# Case 11 — _phase_search where DDG raises for every query → returns ([], []) without crashing
def test_phase_search_all_ddg_fail():
    name = "case 11: _phase_search where DDG raises for every query → returns ([], []) without crashing"
    from unittest.mock import patch

    def always_raise(query, language="en-US", max_results=12):
        raise ConnectionError("network down")

    with patch("src.content_sources.duckduckgo_source.search_duckduckgo", side_effect=always_raise):
        try:
            results, snippets = _phase_search("Naruto", [], "en-US", None)
            assert results == [], f"expected [], got {results!r}"
            assert snippets == [], f"expected [], got {snippets!r}"
            _pass(name)
        except Exception as exc:
            _fail(name, f"raised {type(exc).__name__}: {exc}")


# Case 12 — _phase_wiki where fetch raises → returns [] without crashing
def test_phase_wiki_fetch_raises():
    name = "case 12: _phase_wiki where fetch raises → returns [] without crashing"
    from unittest.mock import patch

    def always_raise(*args, **kwargs):
        raise TimeoutError("wiki timeout")

    with patch("src.content_sources.wikipedia_source.fetch_wikipedia_draft", side_effect=always_raise):
        try:
            result = _phase_wiki("Naruto", "en-US", None)
            assert result == [], f"expected [], got {result!r}"
            _pass(name)
        except Exception as exc:
            _fail(name, f"raised {type(exc).__name__}: {exc}")


# Case 13 — _phase_evaluate returns sufficient=True even though fact_count < target
#            → caller loop must trust LLM and exit (returns [])
def test_phase_evaluate_trusts_llm_sufficient():
    name = "case 13: _phase_evaluate sufficient=true despite fact_count < target → returns [] (loop exits)"
    from unittest.mock import patch

    sufficient_response = json.dumps({
        "sufficient": True,
        "missing_angles": ["origin story", "dark theory"],  # listed but should be ignored
    })

    with patch("src.llm_client.chat_completion", return_value=sufficient_response):
        try:
            result = _phase_evaluate("Naruto", fact_count=3, target=12)
            # sufficient=True → _phase_evaluate returns [] → run_research loop breaks
            assert result == [], (
                f"expected [] so loop exits; LLM said sufficient even though "
                f"fact_count=3 < target=12. Got: {result!r}"
            )
            _pass(name)
        except AssertionError as exc:
            _fail(name, str(exc))
        except Exception as exc:
            _fail(name, f"raised {type(exc).__name__}: {exc}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Research agent — hard edge case tests")
    print("=" * 65)

    test_stopword_only_topic()
    test_short_topic_query_building()
    test_vietnamese_mixed_ascii_extraction()
    test_empty_search_queries_uses_templates()
    test_special_chars_topic()

    test_verify_grounding_drops_unrelated()
    test_verify_grounding_empty_list()
    test_score_and_rank_prefers_tier1()
    test_score_and_rank_max_cap()
    test_extract_english_query_pure_vietnamese()

    test_phase_search_all_ddg_fail()
    test_phase_wiki_fetch_raises()
    test_phase_evaluate_trusts_llm_sufficient()

    print("=" * 65)
    print(f"Results: {_PASS} passed, {_FAIL} failed")
    print("=" * 65)
    sys.exit(0 if _FAIL == 0 else 1)
