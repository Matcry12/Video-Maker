from src.agent.script_citations import extract_citations
from src.agent.contracts import CitationMap


def test_extract_basic_citations():
    script = {
        "blocks": [
            {"text": "Subaru uses Return by Death. [F001] It has a psychological cost. [F002]"},
        ],
    }
    valid = {"F001", "F002"}
    cleaned, cmap = extract_citations(script, valid)
    assert len(cmap.citations) == 2
    assert cmap.citations[0].fact_id == "F001"
    assert cmap.citations[1].fact_id == "F002"
    assert cmap.unused_fact_ids == []
    assert cmap.uncited_sentences == []
    assert "[F001]" not in cleaned["blocks"][0]["text"]
    assert "[F002]" not in cleaned["blocks"][0]["text"]


def test_multi_id_citation():
    script = {"blocks": [{"text": "A claim with two sources. [F001, F003]"}]}
    cleaned, cmap = extract_citations(script, {"F001", "F003"})
    assert {c.fact_id for c in cmap.citations} == {"F001", "F003"}


def test_uncited_sentence_logged():
    # Three sentences: first two uncited, third cited — rate=1/3
    # Split happens at ". S" and ". T" (uppercase after period+space)
    script = {"blocks": [{"text": "First uncited claim. Second uncited claim. Third claim. [F001]"}]}
    cleaned, cmap = extract_citations(script, {"F001"})
    assert len(cmap.uncited_sentences) == 2
    assert cmap.citation_rate == pytest_approx(1 / 3)


def test_invalid_fact_id_ignored():
    script = {"blocks": [{"text": "Hallucinated source. [F999]"}]}
    cleaned, cmap = extract_citations(script, {"F001"})
    assert cmap.citations == []
    assert cmap.uncited_sentences == [(0, 0)]


def test_unused_facts_reported():
    script = {"blocks": [{"text": "Only uses one. [F001]"}]}
    cleaned, cmap = extract_citations(script, {"F001", "F002", "F003"})
    assert set(cmap.unused_fact_ids) == {"F002", "F003"}


def pytest_approx(v, tol=0.01):
    class _Approx:
        def __eq__(self, other): return abs(other - v) < tol
    return _Approx()
