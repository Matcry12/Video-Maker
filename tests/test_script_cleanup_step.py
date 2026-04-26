from __future__ import annotations

from src.agent.script_agent import apply_citation_cleanup


def test_apply_citation_cleanup_strips_visible_fact_markers() -> None:
    script = {
        "blocks": [
            {
                "text": (
                    "Johnny was a prodigy before the fall. [F004] "
                    "He entered the race to cure his legs. [F002, F004]"
                )
            }
        ]
    }
    facts = [
        {"fact_id": "F002", "fact_text": "He entered the race to cure his legs."},
        {"fact_id": "F004", "fact_text": "Johnny was a prodigy before the fall."},
    ]
    warnings: list[str] = []

    cleaned = apply_citation_cleanup(script, facts, warnings)

    assert "[F004]" not in cleaned["blocks"][0]["text"]
    assert "[F002, F004]" not in cleaned["blocks"][0]["text"]
    assert cleaned["__citation_map__"]["citation_rate"] == 1.0
    assert warnings == []


def test_apply_citation_cleanup_skips_when_no_fact_ids_exist() -> None:
    script = {"blocks": [{"text": "No citations here. [F001]"}]}

    cleaned = apply_citation_cleanup(script, [{"fact_text": "missing id"}], [])

    assert cleaned == script
