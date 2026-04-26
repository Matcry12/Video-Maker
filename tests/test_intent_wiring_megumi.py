"""End-to-end regression canary: Megumi intent wiring.

This test proves that user intent survives the full pipeline. It requires live
API keys and is skipped in offline CI. Run with:

    ./.venv/bin/python -m pytest -m live tests/test_intent_wiring_megumi.py -v
"""
import pytest

from src.agent.core import VideoAgent
from src.agent.models import AgentConfig


@pytest.mark.live
def test_megumi_pipeline_covers_cursed_technique_and_sukuna():
    """Pipeline must extract facts about Ten Shadows AND Sukuna possession.

    This canary fails if user intent is lost between stages — i.e., if the
    research extractor ignores the ANGLES TO COVER and returns trivia instead.
    """
    cfg = AgentConfig()
    agent = VideoAgent()
    result = agent.run(
        "Generate a video about Megumi in Jujutsu Kaisen — cover his "
        "cursed technique, comparing with how Sukuna uses his ability "
        "in his body.",
        user_config=cfg,
    )
    assert result.success, f"Pipeline failed: {result.error}"
    facts_text = " ".join(
        f.get("fact_text", "").lower() for f in result.script.get("facts", [])
    )
    # Fallback: check script blocks if facts not in result
    if not facts_text:
        facts_text = " ".join(
            b.get("text", "").lower()
            for b in result.script.get("blocks", [])
        )
    assert any(k in facts_text for k in ("ten shadows", "shikigami", "cursed technique")), (
        f"No cursed-technique facts found. facts_text snippet: {facts_text[:300]!r}"
    )
    assert any(k in facts_text for k in ("sukuna", "possess", "inherit")), (
        f"No Sukuna-possession facts found. facts_text snippet: {facts_text[:300]!r}"
    )
