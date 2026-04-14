"""Script agent — writes narration script, decides image display, supports writing modes."""

import logging
import re
from typing import Any, Callable, Optional

from .models import AgentPlan, ScriptResult

logger = logging.getLogger(__name__)

# Topic categories that use creative writing mode
_CREATIVE_CATEGORIES = frozenset({
    "anime", "entertainment", "gaming", "music", "celebrity",
})

# Keywords that suggest background image display mode
_BG_CONTENT_SIGNALS = frozenset({
    "landscape", "ocean", "mountain", "city skyline", "sunset", "sunrise",
    "forest", "aerial", "panorama", "coastline", "valley", "glacier",
    "desert", "waterfall", "aurora", "skyline",
})

# Keywords that suggest popup image display mode
_POPUP_CONTENT_SIGNALS = frozenset({
    "killed", "died", "built", "discovered", "invented", "arrested",
    "billion", "million", "percent", "government", "collapsed", "exploded",
    "investigation", "trial", "scandal", "war", "battle", "crisis",
})


def run_script(
    facts: list[dict],
    plan: AgentPlan,
    emit: Optional[Callable[[dict], None]] = None,
) -> ScriptResult:
    """Write a narration script from facts, lint it, and decide image display.

    Flow:
    1. Choose writing mode (factual vs creative) based on topic_category
    2. Write script (1 Groq call)
    3. Lint (deterministic, 0 Groq calls)
    4. If lint fails: rewrite with specific lint feedback (1 more Groq call)
    5. Decide image display mode
    """
    warnings: list[str] = []

    def _emit(phase: str, message: str, **extra):
        if emit:
            try:
                emit({"phase": phase, "message": message, **extra})
            except Exception:
                pass

    # --- CHOOSE WRITING MODE ---
    topic_category = getattr(plan, "topic_category", "")
    writing_mode = "creative" if topic_category in _CREATIVE_CATEGORIES else "factual"

    # Build style hint based on writing mode
    style = plan.style or None
    if writing_mode == "creative" and not style:
        style = "creative"

    # --- SELECT SKILL ---
    from .skill_selector import select_skill

    selected_skill = select_skill(plan)
    _emit("script", f"Using skill: {selected_skill.get('name', 'General')}")
    logger.info("Script skill: %s", selected_skill.get("skill_id", "_default"))

    # --- WRITE ---
    _emit("script", f"Writing {writing_mode} narration script...")

    from ..content_sources.fact_script_writer import write_script_from_facts

    script: dict[str, Any] = {}
    writer_meta: dict[str, Any] = {}

    try:
        writer_result = write_script_from_facts(
            facts,
            language=plan.language,
            target_blocks=plan.target_blocks,
            style=style,
            skill=selected_skill,
        )
        script = writer_result["script"]
        writer_meta = writer_result.get("meta", {})

        # Validate: if language is Vietnamese but script came back in English, reject it
        if plan.language.startswith("vi"):
            blocks = script.get("blocks", [])
            english_blocks = 0
            for b in blocks:
                text = b.get("text", "")
                # Check if text is mostly ASCII (English) — Vietnamese has many non-ASCII chars
                if text and sum(1 for c in text if ord(c) < 128) / max(len(text), 1) > 0.95:
                    english_blocks += 1
            if english_blocks > len(blocks) / 2:
                logger.warning(
                    "Script writer returned %d/%d English blocks for Vietnamese target. Rejecting.",
                    english_blocks, len(blocks),
                )
                raise ValueError(
                    f"Script is in English ({english_blocks}/{len(blocks)} blocks) "
                    f"but target language is {plan.language}. Facts may be in wrong language."
                )
    except Exception as exc:
        warnings.append(f"Script writer failed: {exc}. Using direct assembly.")
        logger.warning("Script writer failed: %s", exc)
        # Fallback: combine all facts into ONE single continuous block
        is_vietnamese = plan.language.startswith("vi")
        fact_texts = []
        for f in facts[:plan.target_blocks]:
            text = f.get("fact_text", "").strip()
            # Strip markdown formatting
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            text = re.sub(r"\*(.+?)\*", r"\1", text)
            text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
            text = re.sub(r"^[A-ZÀ-Ỹ][^:]{2,30}:\s*", "", text).strip()
            text = text.replace("`", "")
            # Skip English-only facts for Vietnamese TTS
            if is_vietnamese and text and all(ord(c) < 256 for c in text.replace(" ", "")):
                logger.warning("Skipping English-only fact in Vietnamese mode: '%s...'", text[:50])
                continue
            if text:
                fact_texts.append(text)
        # Build single continuous block
        hook = f"Đây là {plan.target_blocks} sự thật thú vị về {plan.topic}!" if is_vietnamese else f"Here are {plan.target_blocks} interesting facts about {plan.topic}!"
        if not fact_texts:
            if is_vietnamese:
                fact_texts.append(f"{plan.topic} là một chủ đề rất thú vị. Hãy cùng tìm hiểu thêm nhé!")
            warnings.append("All research facts were English — could not use them for Vietnamese TTS.")
        full_text = hook + " " + " ".join(fact_texts)
        script = {
            "language": plan.language,
            "blocks": [{"text": full_text}],
        }

    # --- LINT ---
    _emit("script", "Quality checking script...")

    from ..content_sources import lint_script

    lint_result = lint_script(script)

    # --- REWRITE WITH FEEDBACK (max 1 attempt) ---
    if lint_result["status"] in ("soft_fail", "hard_fail") and lint_result.get("issues"):
        # Build specific feedback from lint issues
        feedback_lines = _build_lint_feedback(lint_result["issues"])
        original_score = lint_result["score"]

        warnings.append(
            f"Script lint {lint_result['status']} (score={original_score}). "
            f"Rewriting with {len(feedback_lines)} specific issues..."
        )
        _emit(
            "script",
            f"Script quality low (score={original_score}). "
            f"Rewriting with specific feedback...",
        )

        try:
            # Pass lint issues as extra constraints to the writer
            writer_result2 = write_script_from_facts(
                facts,
                language=plan.language,
                target_blocks=plan.target_blocks,
                style=style,
                extra_constraints=feedback_lines,
                skill=selected_skill,
            )
            script2 = writer_result2["script"]
            lint_result2 = lint_script(script2)

            # Keep whichever version scores better
            if lint_result2["score"] >= original_score:
                script = script2
                lint_result = lint_result2
                writer_meta = writer_result2.get("meta", {})
                _emit("script", f"Rewrite improved score: {original_score} → {lint_result2['score']}")
            else:
                warnings.append(
                    f"Rewrite scored lower ({lint_result2['score']} < {original_score}). "
                    "Keeping original."
                )
                _emit("script", "Rewrite didn't improve — keeping original.")
        except Exception as exc:
            warnings.append(f"Rewrite failed: {exc}")

    # --- DECIDE IMAGE DISPLAY ---
    image_display = plan.image_display  # default from planner

    # Only override if user didn't explicitly set it
    user_set_display = (
        plan.user_overrides is not None
        and plan.user_overrides.image_display is not None
    )
    if not user_set_display:
        image_display = _infer_image_display(script, plan.style)

    script["image_mode"] = image_display
    if plan.bgm_mood:
        script["bgm_mood"] = plan.bgm_mood

    writer_meta["writing_mode"] = writing_mode
    writer_meta["skill_id"] = selected_skill.get("skill_id", "_default")
    writer_meta["skill_name"] = selected_skill.get("name", "General")

    return ScriptResult(
        script=script,
        image_display=image_display,
        lint_score=lint_result.get("score", 0),
        lint_status=lint_result.get("status", "pass"),
        writer_meta=writer_meta,
        warnings=warnings,
    )


def _build_lint_feedback(issues: list[dict]) -> list[str]:
    """Convert lint issues into specific rewrite instructions for the LLM."""
    feedback: list[str] = []
    for issue in issues:
        code = issue.get("code", "")
        detail = issue.get("detail", "")
        block_idx = issue.get("block_index")

        if code == "GENERIC_OPENER" or code == "VN_GENERIC_OPENER":
            feedback.append(
                f"Block 0 (hook) starts with a generic opener. "
                "Rewrite to grab attention in the first 2 seconds — "
                "use a startling fact, bold claim, or vivid image."
            )
        elif code == "WEAK_ENDING" or code == "VN_WEAK_ENDING":
            feedback.append(
                f"Last block ends weakly. "
                "Rewrite the ending with emotional resonance, a callback to the hook, "
                "or an unexpected final detail."
            )
        elif code == "REPEATED_IMPACT":
            feedback.append(f"Fix: {detail}")
        elif code == "REPEATED_ADJECTIVE" or code == "VN_DRAMATIC_ADJECTIVE":
            feedback.append(f"Fix: {detail}")
        elif code == "LOW_BLOCK_NOVELTY":
            feedback.append(
                f"Blocks {block_idx} and {block_idx + 1} are too similar. "
                "Make each block add new information or escalate stakes."
            )
        elif code == "HIGH_AVG_SENTENCE_LENGTH":
            feedback.append(
                "Sentences are too long for voice-over. "
                "Use shorter, punchier sentences. Alternate short and long."
            )
        elif code in ("TOO_FEW_BLOCKS", "TOO_SHORT"):
            feedback.append(f"Fix: {detail}")

    return feedback


def _infer_image_display(script: dict, style: str) -> str:
    """Decide popup vs background based on actual script content."""
    texts = " ".join(
        b.get("text", "") for b in script.get("blocks", []) if isinstance(b, dict)
    ).lower()

    bg_score = sum(1 for kw in _BG_CONTENT_SIGNALS if kw in texts)
    popup_score = sum(1 for kw in _POPUP_CONTENT_SIGNALS if kw in texts)

    # Cinematic style leans toward background
    if style in ("cinematic",):
        bg_score += 2

    if bg_score > popup_score:
        return "background"
    return "popup"
