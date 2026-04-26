import os
import pytest

from src.agent.research_agent import run_research
from src.agent.entity_sanitizer import is_contaminated, forbidden_entities


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_rezero_no_al_character():
    """The Re:Zero research must not produce facts mentioning a character 'Al'
    that does not exist in the canonical material."""
    result = run_research(
        topic="Re:Zero dark secrets",
        search_queries=[
            "Re:Zero light novel canon",
            "Subaru Natsuki Return by Death origin",
        ],
        language="en-US",
        skill_id="dark_secrets",
        topic_aliases=["Re:Zero", "Re Zero", "Rezero", "Re:Zero kara Hajimeru Isekai Seikatsu"],
    )
    facts = result["facts"]
    assert len(facts) >= 3, f"Too few facts: {len(facts)}"
    for f in facts:
        text = f["fact_text"].lower()
        assert "al uses" not in text, f"Hallucinated fact: {f['fact_text']}"
        assert "al and subaru" not in text, f"Hallucinated fact: {f['fact_text']}"


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_rezero_no_jjk_contamination():
    """No research fact should reference Jujutsu Kaisen characters."""
    fb = forbidden_entities("Re:Zero", topic_aliases=["Re Zero", "Rezero"])
    result = run_research(
        topic="Re:Zero dark secrets",
        search_queries=["Re:Zero light novel"],
        language="en-US",
        skill_id="dark_secrets",
        topic_aliases=["Re:Zero", "Re Zero", "Rezero"],
    )
    for f in result["facts"]:
        contam, hits = is_contaminated(f["fact_text"], fb)
        assert not contam, f"Contamination in fact: {f['fact_text']!r} hits={hits}"
