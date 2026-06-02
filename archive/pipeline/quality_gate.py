"""Quality gate: deterministic checks + LLM judge.

Returns GateResult with a combined score. Pipeline shall:
  - treat passed=False as a signal to rewrite once
  - never silently skip the gate (skipped=True is still recorded)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from src.agent.gate_deterministic import run_deterministic_checks
from src.agent.robust_json import extract_json_dict, parse_yes_no

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    passed: bool
    det_score: int = 0         # 0..60
    llm_yes: int = 0           # 0..5
    llm_score: int = 0         # 0..50
    combined_score: int = 0    # 0..100
    det_issues: list[str] = field(default_factory=list)
    llm_feedback: str = ""
    skipped: bool = False


_DEFAULT_QUESTIONS = [
    "LORE vs RECAP: Does the script reveal hidden details/theories/dark secrets rather than basic plot summary?",
    "SURPRISE: Is there at least ONE moment where a knowledgeable fan would think 'I didn't know that'?",
    "NATURAL VOICE: Does this sound like a hyped fan explaining to a friend, NOT a wiki article being read aloud?",
    "SENTENCE RHYTHM: Are sentence lengths varied (mixing short punches with longer flowing lines), or do they all feel the same length?",
    "SPOKEN-NOT-WRITTEN: Is the script free of essay/forum vocab and flat reporting (e.g. 'It said X', 'It is Y', 'Furthermore...') that nobody says aloud?",
]


def _load_skill_questions(skill_id: str) -> list[str]:
    if not skill_id:
        return _DEFAULT_QUESTIONS
    try:
        from pathlib import Path
        import json
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / "skills" / f"{skill_id}.json"
        if not path.exists():
            return _DEFAULT_QUESTIONS
        data = json.loads(path.read_text(encoding="utf-8"))
        qs = data.get("quality_gate_questions")
        if isinstance(qs, list) and len(qs) == 5 and all(isinstance(q, str) for q in qs):
            return qs
        logger.warning("skill %r has %d quality_gate_questions (expected 5) — using defaults",
                       skill_id, len(qs) if isinstance(qs, list) else 0)
    except Exception as exc:
        logger.debug("load skill questions failed: %s", exc)
    return _DEFAULT_QUESTIONS


_SYSTEM_TEMPLATE = (
    "You are a brutally honest YouTube Shorts script evaluator optimized for "
    "viewer retention and algorithm performance.\n\n"
    "CALIBRATION RULES — read before answering:\n"
    "  - Default to 'no'. Only answer 'yes' when the criterion is CLEARLY met.\n"
    "  - 'Pretty good' is NOT 'yes'. Fail decent-but-not-great scripts.\n"
    "  - If your weakest_part complaint would contradict a 'yes' answer, that\n"
    "    'yes' is wrong — change it to 'no'.\n"
    "  - For any 'no', the weakest_part MUST quote the offending sentence in\n"
    "    double quotes so a human can verify your call.\n\n"
    "Respond ONLY with strict JSON — no markdown, no prose, no explanation:\n"
    '{"q1":"yes"|"no","q2":"yes"|"no","q3":"yes"|"no","q4":"yes"|"no","q5":"yes"|"no","weakest_part":"one short sentence; quote the offending text in double quotes"}'
)


def _build_user_prompt(script: dict, topic: str, questions: list[str]) -> str:
    blocks = script.get("blocks") or []
    joined = "\n\n".join(f"[B{i}] {b.get('text','')}" for i, b in enumerate(blocks))
    q_lines = "\n".join(f"q{i+1}: {q}" for i, q in enumerate(questions))
    return (
        f"TOPIC: {topic}\n\n"
        f"SCRIPT:\n{joined}\n\n"
        f"EVALUATE EACH QUESTION:\n{q_lines}\n\n"
        "Return the JSON now."
    )


def run_quality_gate(script: dict, plan=None, facts=None, skill_id: str = "") -> GateResult:
    # -------- Deterministic pass (always runs, no API needed) -----------
    if plan is not None:
        det_score, det_issues = run_deterministic_checks(script, plan)
    else:
        det_score, det_issues = 0, ["plan unavailable"]

    # -------- LLM pass (skipped if no API key) --------------------------
    llm_yes = 0
    llm_feedback = ""
    skipped = False
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        skipped = True
        logger.warning("quality_gate: no API key set — deterministic only")
    else:
        questions = _load_skill_questions(skill_id)
        topic = getattr(plan, "topic", "") if plan else script.get("topic", "")
        system = _SYSTEM_TEMPLATE
        user = _build_user_prompt(script, topic, questions)
        try:
            from src.llm_client import chat_completion
            raw = chat_completion(
                system=system, user=user, stage="quality_gate",
                temperature=0.3, timeout=25.0,
            )
            data = extract_json_dict(raw) or {}
            llm_yes = sum(1 for k in ("q1", "q2", "q3", "q4", "q5")
                          if parse_yes_no(data.get(k)))
            llm_feedback = str(data.get("weakest_part") or "").strip()
        except Exception as exc:
            logger.warning("quality_gate LLM call failed: %s", exc)
            skipped = True

    llm_score = llm_yes * 10
    combined = det_score + llm_score

    # Pass rules (all must hold):
    #   det_score >= 45   (at most one 10-pt check can fully fail)
    #   llm_yes  >= 3    (majority of judge questions pass) OR gate was skipped
    #   combined >= 75   (was 88 when must_cover gave a free +10; rebalanced
    #                    after replacing it with stricter natural-speech check)
    llm_ok = (llm_yes >= 3) or skipped
    passed = (det_score >= 45) and llm_ok and (combined >= 75 or skipped)

    return GateResult(
        passed=passed,
        det_score=det_score,
        llm_yes=llm_yes,
        llm_score=llm_score,
        combined_score=combined,
        det_issues=det_issues,
        llm_feedback=llm_feedback,
        skipped=skipped,
    )
