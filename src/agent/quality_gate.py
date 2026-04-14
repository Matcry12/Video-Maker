"""Quality Gate — quick LLM check before expensive rendering."""

import json
import logging
import os
import re
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a brutally honest YouTube Shorts script evaluator optimized for viewer retention and algorithm performance. Answer five yes/no questions about the script, then identify the weakest part.

Questions:
1. HOOK: Does the FIRST sentence contain a shocking fact, bold claim, or curiosity gap that would stop a viewer from scrolling? (Reject if it starts with character introduction, plot summary, or "In [year]...")
2. LORE vs RECAP: Does the script reveal HIDDEN details, theories, or dark secrets — or is it just a basic plot summary that any fan already knows?
3. SURPRISE: Is there at least one moment where even a knowledgeable fan would think "I didn't know that!"?
4. SEAMLESS LOOP: Does the last sentence connect back to the first sentence, creating a natural replay loop? (The ending should feel like a cliffhanger, not a dead-end conclusion.)
5. PACING: Does the script escalate tension — each revelation more shocking than the last — or does it feel flat and monotonous?

Respond ONLY with strict JSON — no markdown, no explanation:
{"q1": "yes"|"no", "q2": "yes"|"no", "q3": "yes"|"no", "q4": "yes"|"no", "q5": "yes"|"no", "weakest_part": "..."}
"""


def run_quality_gate(
    script: dict[str, Any],
    topic: str,
) -> dict[str, Any]:
    """Evaluate script quality with 3 yes/no questions.

    Returns {"passed": bool, "yes_count": int, "feedback": str, "skipped": bool}

    - 2-3 "yes" → PASS
    - 0-1 "yes" → FAIL with feedback for rewrite
    - No API key or error → skipped=True, passed=True (don't block rendering)
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.info("quality_gate: no GROQ_API_KEY — skipping")
        return {"passed": True, "yes_count": 0, "feedback": "", "skipped": True}

    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    blocks = script.get("blocks") or script.get("segments") or []
    script_text = " ".join(
        b.get("text", "") for b in blocks if isinstance(b, dict)
    ).strip()
    if not script_text:
        logger.info("quality_gate: empty script text — skipping")
        return {"passed": True, "yes_count": 0, "feedback": "", "skipped": True}

    user_message = f"Topic: {topic}\n\nScript:\n{script_text}"

    try:
        from ..llm_client import chat_completion

        raw = chat_completion(
            system=_SYSTEM_PROMPT,
            user=user_message,
            model=model,
            temperature=0.3,
            timeout=20.0,
        )

        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        # Extract first JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in response: {raw!r}")

        data = json.loads(match.group())
        yes_count = sum(
            1 for key in ("q1", "q2", "q3", "q4", "q5") if str(data.get(key, "")).lower() == "yes"
        )
        passed = yes_count >= 3
        feedback = data.get("weakest_part", "")

        logger.info("quality_gate: yes_count=%d passed=%s", yes_count, passed)
        return {"passed": passed, "yes_count": yes_count, "feedback": feedback, "skipped": False}

    except Exception as exc:  # noqa: BLE001
        logger.warning("quality_gate: error (%s) — skipping gate", exc)
        return {"passed": True, "yes_count": 0, "feedback": "", "skipped": True}
