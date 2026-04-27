"""Script agent — writes narration script, decides image display, supports writing modes."""

import logging
import re
from typing import Any, Callable, Optional

from .models import AgentPlan, ScriptResult
from ..agent_config import load_agent_settings

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

_BRAINROT_PACING = (
    "MAX 8 words per sentence. Use single-word sentences for impact: 'Wait.' 'No.' 'Gone.' 'Insane.'\n"
    "For EVERY fact follow this 3-line pattern:\n"
    "  1. The fact (4-8 words)\n"
    "  2. One reaction word or short question (1-3 words)\n"
    "  3. A follow-up consequence or deeper detail (4-8 words)\n"
    "Example: 'He never learned healing. Impossible. His own power does it for him.'\n"
    "Do this for every fact. Never move to the next fact after just one sentence."
)

_BRAINROT_TONE = (
    "Your hype friend texting insane facts at 2am. Zero academic language. "
    "Zero explanations. Pure shock, one fact after another."
)

_BRAINROT_INJECTION = (
    "BRAINROT FORMAT — CRITICAL RULES:\n"
    "- Sentences: MAX 8 words. Shorter is better.\n"
    "- BANNED phrases: 'this connects to', 'the reason is', 'which means', "
    "'and if we trace', 'the reason this matters', 'it is worth noting'.\n"
    "- Assert facts bluntly. Never explain why something is surprising — just say the thing.\n"
    "- Use single-word sentences for punch: 'Wait.' 'No.' 'Gone.' 'Insane.'\n"
    "- Use ONLY facts from the research. When all facts are covered, STOP. Never invent filler.\n"
    "- Do NOT pad with vague statements like 'his power grows', 'the future is uncertain'.\n"
    "- The viewer already knows the characters. Skip all background intro."
)


def _apply_brainrot_overlay(skill: dict) -> dict:
    """Inject brainrot pacing/tone on top of any content skill's structure.

    Reads rules from skills/brainrot.json so overlay always stays in sync
    with the brainrot skill definition.
    """
    import copy
    import json
    from pathlib import Path as _Path
    brainrot_path = _Path(__file__).parent.parent.parent / "skills" / "brainrot.json"
    brainrot = json.loads(brainrot_path.read_text(encoding="utf-8"))

    skill = copy.deepcopy(skill)
    structure = skill.setdefault("structure", {})
    brainrot_structure = brainrot.get("structure", {})
    structure["tone"] = brainrot_structure.get("tone", "")
    structure["pacing_rule"] = brainrot_structure.get("pacing_rule", "")
    skill["prompt_injection"] = (brainrot.get("prompt_injection", "") or "").strip()
    return skill


def _clean_script_text(script: dict[str, Any]) -> dict[str, Any]:
    """Strip LLM formatting artifacts from block text (spaces before punctuation, double spaces)."""
    for block in script.get("blocks", []):
        text = block.get("text", "")
        if text:
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)
            text = re.sub(r' {2,}', ' ', text)
            block["text"] = text.strip()
    return script


def apply_citation_cleanup(
    script: dict[str, Any],
    facts: list[dict],
    warnings: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Strip inline fact markers from visible text and attach citation metadata."""
    warnings = warnings if warnings is not None else []

    from .script_citations import extract_citations

    valid_ids: set[str] = set()
    for fact in facts or []:
        fid = fact.get("fact_id") if isinstance(fact, dict) else None
        if fid:
            valid_ids.add(str(fid).upper())

    if not valid_ids:
        return script

    cleaned_script, citation_map = extract_citations(script, valid_ids)
    cleaned_script["__citation_map__"] = {
        "citations": [
            {"fact_id": c.fact_id, "block_idx": c.block_idx, "sentence_idx": c.sentence_idx}
            for c in citation_map.citations
        ],
        "unused_fact_ids": citation_map.unused_fact_ids,
        "uncited_sentences": [list(s) for s in citation_map.uncited_sentences],
        "citation_rate": citation_map.citation_rate,
    }
    if citation_map.citation_rate < 0.6:
        warnings.append(
            f"Script citation rate low: {citation_map.citation_rate:.0%} "
            f"({len(citation_map.uncited_sentences)} sentences not grounded)"
        )

    return cleaned_script


def run_script(
    facts: list[dict],
    plan: AgentPlan,
    emit: Optional[Callable[[dict], None]] = None,
    knowledge_doc: str = "",
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

    forced_skill = plan.user_overrides.skill_id if plan.user_overrides else None
    is_brainrot = plan.style == "brainrot"

    if is_brainrot and forced_skill != "brainrot":
        # Select content skill without "brainrot" in the BM25 query (it would
        # match the standalone brainrot skill instead of the content skill)
        _plan_no_style = plan.model_copy(update={"style": ""})
        selected_skill = select_skill(_plan_no_style, forced_skill_id=forced_skill)
        selected_skill = _apply_brainrot_overlay(selected_skill)
    else:
        selected_skill = select_skill(plan, forced_skill_id=forced_skill)

    _emit("script", f"Using skill: {selected_skill.get('name', 'General')}{' [brainrot]' if is_brainrot else ''}")
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
            video_goal=plan.user_prompt or plan.topic,
            must_cover=", ".join(plan.must_cover or []),
            knowledge_doc=knowledge_doc,
        )
        script = writer_result["script"]
        writer_meta = writer_result.get("meta", {})
        script = apply_citation_cleanup(script, facts, warnings)
        script = _clean_script_text(script)

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
        # Build single continuous block — use skill example hook, never generic "Here are N facts"
        hook_key = "example_hook_vi" if is_vietnamese else "example_hook_en"
        hook = selected_skill.get(hook_key, "") or selected_skill.get("example_hook_en", "")
        hook = hook.replace("[topic]", plan.topic)
        if not hook:
            first_fact = fact_texts[0][:120] if fact_texts else ""
            hook = f"{plan.topic} — {first_fact}." if first_fact else plan.topic
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

    # --- REWRITE WITH FEEDBACK (loop up to max_lint_rewrites attempts) ---
    _agent_settings = load_agent_settings()
    max_lint_rewrites = int(_agent_settings.get("max_lint_rewrites", 3))
    lint_accept_score = int(_agent_settings.get("lint_accept_score", 80))

    if lint_result["status"] in ("soft_fail", "hard_fail") and lint_result.get("issues"):
        best_script = script
        best_lint = lint_result
        best_meta = writer_meta

        for attempt in range(1, max_lint_rewrites + 1):
            if best_lint["score"] >= lint_accept_score:
                break
            if best_lint["status"] not in ("soft_fail", "hard_fail"):
                break

            feedback_lines = _build_lint_feedback(best_lint["issues"])
            warnings.append(
                f"Script lint {best_lint['status']} (score={best_lint['score']}, attempt={attempt}). "
                f"Rewriting with {len(feedback_lines)} specific issues..."
            )
            _emit(
                "script",
                f"Script quality low (score={best_lint['score']}, attempt {attempt}/{max_lint_rewrites}). "
                f"Rewriting with specific feedback...",
            )

            try:
                writer_result2 = write_script_from_facts(
                    facts,
                    language=plan.language,
                    target_blocks=plan.target_blocks,
                    style=style,
                    extra_constraints=feedback_lines,
                    skill=selected_skill,
                    video_goal=plan.user_prompt or plan.topic,
                    must_cover=", ".join(plan.must_cover or []),
                    knowledge_doc=knowledge_doc,
                )
                script2 = writer_result2["script"]
                lint_result2 = lint_script(script2)

                if lint_result2["score"] > best_lint["score"]:
                    _emit(
                        "script",
                        f"Rewrite improved score: {best_lint['score']} → {lint_result2['score']}",
                    )
                    best_script = script2
                    best_lint = lint_result2
                    best_meta = writer_result2.get("meta", {})
                else:
                    warnings.append(
                        f"Rewrite attempt {attempt} scored {lint_result2['score']} "
                        f"(best={best_lint['score']}). Keeping best so far."
                    )
                    _emit("script", f"Rewrite attempt {attempt} didn't improve — keeping best so far.")
            except Exception as exc:
                warnings.append(f"Rewrite attempt {attempt} failed: {exc}")

        script = best_script
        lint_result = best_lint
        writer_meta = best_meta

    # --- PHRASE WINDOWS (per-moment image variety) ---
    blocks = script.get("blocks", [])
    if blocks and blocks[0].get("text"):
        try:
            from ..images.phrase_windows import split_into_windows
            # Pick shortest alias ≤8 chars as franchise tag (e.g. "JJK" not "Toji Fushiguro")
            franchise = ""
            for alias in (plan.entity_aliases or []):
                if alias and len(alias.strip()) <= 8:
                    franchise = alias.strip()
                    break
            blocks[0]["windows"] = split_into_windows(blocks[0]["text"], plan.topic, franchise=franchise)
        except Exception as exc:
            warnings.append(f"Phrase window split failed: {exc}")

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

    if popup_score > bg_score:
        return "popup"
    return "background"
