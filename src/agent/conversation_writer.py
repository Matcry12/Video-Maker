"""Conversation script writer — generates a two-character dialogue script.

Characters: Marcus (Skeptic) and Maya (Explainer).
Output matches the conversation_script.json schema used by the long-form
render pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.llm_client import chat_completion_with_meta
from src.agent.robust_json import extract_first_json

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"

_REQUIRED_YOUTUBE_KEYS = ("title", "description", "chapters", "hashtags", "tags")


def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def write_conversation_script(topic: str, facts_text: str) -> dict:
    """Call LLM to generate a conversation_script dict for topic + facts.

    Returns a dict matching the conversation_script.json schema:
    {
      "topic": str,
      "language": "en-US",
      "sections": [
        {
          "title": str,
          "image_keywords": list[str],
          "mood": str,
          "turns": [
            {"speaker": "maya"|"marcus", "line": str, "exag": float}
          ]
        }
      ],
      "youtube": {
        "title": str, "description": str, "chapters": list[str],
        "hashtags": list[str], "tags": list[str]
      }
    }

    Raises ValueError if LLM output doesn't parse or fails validation.
    """
    system_prompt = _load_prompt("conversation_script.txt")

    response = chat_completion_with_meta(
        system=system_prompt,
        user=f"TOPIC: {topic}\n\nRESEARCH FACTS:\n{facts_text}",
        stage="script",
        temperature=0.7,
        max_tokens=8192,
    )
    raw = response.text

    parsed = extract_first_json(raw)
    if parsed is None or not isinstance(parsed, dict):
        logger.warning(
            "conversation_writer: failed to parse JSON from LLM output. "
            "First 500 chars: %s",
            raw[:500],
        )
        raise ValueError(
            f"LLM output for topic {topic!r} did not contain a valid JSON object."
        )

    # --- validate sections ---
    if "sections" not in parsed or not isinstance(parsed["sections"], list):
        raise ValueError("conversation_script: missing or invalid 'sections' list.")

    section_count = len(parsed["sections"])
    if not (4 <= section_count <= 10):
        raise ValueError(
            f"conversation_script: expected 4–10 sections, got {section_count}."
        )

    total_turns = 0
    for i, section in enumerate(parsed["sections"]):
        if "turns" not in section or not isinstance(section["turns"], list):
            raise ValueError(
                f"conversation_script: section {i} missing 'turns' list."
            )
        for turn in section["turns"]:
            # clamp / default exag
            if "exag" not in turn:
                turn["exag"] = 0.6
            else:
                turn["exag"] = max(0.3, min(1.0, float(turn["exag"])))

            speaker = turn.get("speaker", "")
            if speaker not in {"marcus", "maya"}:
                raise ValueError(
                    f"conversation_script: invalid speaker {speaker!r} in section {i}."
                )

        total_turns += len(section["turns"])

    # --- inject missing top-level keys ---
    if "topic" not in parsed:
        parsed["topic"] = topic

    if "language" not in parsed:
        parsed["language"] = "en-US"

    if "youtube" not in parsed or not isinstance(parsed.get("youtube"), dict):
        parsed["youtube"] = {
            "title": "",
            "description": "",
            "chapters": [],
            "hashtags": [],
            "tags": [],
        }
    else:
        yt = parsed["youtube"]
        for key in _REQUIRED_YOUTUBE_KEYS:
            if key not in yt:
                yt[key] = [] if key in ("chapters", "hashtags", "tags") else ""

    logger.info(
        "conversation_writer: topic=%r sections=%d total_turns=%d provider=%s model=%s",
        topic,
        section_count,
        total_turns,
        response.provider,
        response.model,
    )

    return parsed
