"""Video generation agent — autonomous pipeline orchestrator."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .editor_agent import run_editor, run_editor_lab
from .image_agent import run_images
from .models import AgentConfig, AgentPhase, AgentPlan, AgentResult, AgentState
from .plan_agent import plan_from_prompt
from .quality_gate import run_quality_gate
from .research_agent import run_research
from .rag_research_agent import run_agentic_research
from .script_agent import run_script

logger = logging.getLogger(__name__)


class VideoAgent:
    """Autonomous video generation agent.

    Takes a free-form prompt and produces a complete video by orchestrating
    six agent phases: Plan, Research, Script, Quality Gate, Image, Editor.

    Usage:
        agent = VideoAgent()
        result = agent.run("Make a video about the Chernobyl disaster")
        print(result.video_path)
    """

    def __init__(
        self,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.progress_callback = progress_callback
        self.state = AgentState()

    def _emit(self, payload: dict):
        """Send progress update to callback."""
        if not self.progress_callback:
            return
        try:
            self.progress_callback(payload)
        except Exception:
            pass

    def run(
        self,
        prompt: str,
        output_name: str = "",
        user_config: Optional[AgentConfig] = None,
    ) -> AgentResult:
        """Run the full agent pipeline from prompt to video.

        Args:
            prompt: User's free-form request.
            output_name: Optional filename for the output video.
            user_config: Optional explicit config overrides from the user.

        Never raises — returns AgentResult with success=False on failure.
        """
        try:
            return self._run_pipeline(prompt, output_name, user_config)
        except Exception as exc:
            logger.exception("Agent pipeline failed: %s", exc)
            return AgentResult(
                success=False,
                error=str(exc),
                warnings=self.state.warnings,
                script=self.state.script,
            )

    def _run_pipeline(
        self,
        prompt: str,
        output_name: str,
        user_config: Optional[AgentConfig],
    ) -> AgentResult:
        all_warnings: list[str] = []

        # === PHASE 1: PLAN ===
        self.state.phase = AgentPhase.PLAN
        self._emit({"phase": "plan", "message": "Analyzing your request..."})

        plan = plan_from_prompt(prompt, user_config=user_config)
        from .entity_sanitizer import forbidden_entities as _forbidden_entities
        plan.forbidden_entities = _forbidden_entities(plan.topic, topic_aliases=plan.entity_aliases)
        self.state.plan = plan
        logger.info(
            "Agent plan: topic='%s', language='%s', style='%s', "
            "category='%s', blocks=%d, image_display='%s'",
            plan.topic, plan.language, plan.style,
            plan.topic_category, plan.target_blocks, plan.image_display,
        )

        # Generate unique output name with timestamp to avoid overwriting
        run_ts = int(time.time())
        base_name = _sanitize_filename(plan.topic)
        if not output_name:
            output_name = f"{base_name}_{run_ts}"

        # Create run folder for tracking all LLM artifacts
        run_dir = Path(__file__).parent.parent.parent / "output" / "runs" / output_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save plan
        _save_artifact(run_dir, "plan.json", {
            "prompt": prompt,
            "user_prompt": plan.user_prompt,
            "topic": plan.topic,
            "language": plan.language,
            "style": plan.style,
            "topic_category": plan.topic_category,
            "target_blocks": plan.target_blocks,
            "image_display": plan.image_display,
            "search_queries": plan.search_queries,
            "entity_aliases": plan.entity_aliases,
            "must_cover": plan.must_cover,
            "entity_cards": plan.entity_cards,
            "narrative_dynamic": plan.narrative_dynamic,
            "timestamp": run_ts,
        })

        # === PHASE 2: RESEARCH ===
        self.state.phase = AgentPhase.RESEARCH
        self._emit({"phase": "research", "message": f"Researching '{plan.topic}'..."})

        from ..agent_config import load_agent_settings as _load_settings
        _research_mode = _load_settings().get("research_mode", "classic")
        _research_fn = run_agentic_research if _research_mode == "agentic" else run_research

        research_result = _research_fn(
            topic=plan.topic,
            search_queries=plan.search_queries,
            language=plan.language,
            skill_id=(plan.user_overrides.skill_id or '') if plan.user_overrides else '',
            topic_aliases=plan.entity_aliases,
            emit=self._emit,
            user_prompt=plan.user_prompt,
            must_cover=plan.must_cover,
            topic_category=plan.topic_category,
        )
        all_warnings.extend(research_result.get("warnings", []))

        facts = research_result.get("facts", [])
        sources_used = research_result.get("sources_used", [])

        if not facts:
            return AgentResult(
                success=False,
                error=f"No usable facts found for topic '{plan.topic}'.",
                sources_used=sources_used,
                warnings=all_warnings,
            )

        # Save research results
        _save_artifact(run_dir, "research.json", {
            "facts": facts,
            "sources_used": sources_used,
            "warnings": research_result.get("warnings", []),
            "knowledge_doc": research_result.get("coverage", ""),
        })

        # === PHASE 3: SCRIPT ===
        self.state.phase = AgentPhase.SCRIPT
        script_result = run_script(facts, plan, emit=self._emit)
        all_warnings.extend(script_result.warnings)
        self.state.script = script_result.script

        # === PHASE 4: QUALITY GATE ===
        self.state.phase = AgentPhase.QUALITY_GATE
        self._emit({"phase": "quality_gate", "message": "Evaluating script quality..."})

        skill_id = script_result.writer_meta.get("skill_id", "") if script_result.writer_meta else ""
        gate = run_quality_gate(script_result.script, plan=plan, facts=facts, skill_id=skill_id)

        if gate.skipped:
            self._emit({
                "phase": "quality_gate",
                "message": f"QA partial (LLM skipped) det={gate.det_score}/50",
                "det_score": gate.det_score,
                "skipped": True,
            })
        else:
            self._emit({
                "phase": "quality_gate",
                "message": f"QA det={gate.det_score}/50 llm={gate.llm_score}/50 ({gate.combined_score}/100)",
                "det_score": gate.det_score,
                "llm_score": gate.llm_score,
                "combined_score": gate.combined_score,
                "passed": gate.passed,
            })

        if not gate.passed and not gate.skipped:
            # Fail → one targeted rewrite with feedback
            feedback_parts = list(gate.det_issues) + ([gate.llm_feedback] if gate.llm_feedback else [])
            feedback = " | ".join(p for p in feedback_parts if p)
            if feedback:
                self._emit({"phase": "quality_gate", "message": "Script needs improvement, rewriting..."})
                all_warnings.append(
                    f"Quality gate failed (det={gate.det_score}/50 llm={gate.llm_score}/50). "
                    f"Rewriting with feedback: {feedback[:160]}"
                )

                # Re-run script agent with quality gate feedback as extra constraint
                from .script_agent import apply_citation_cleanup
                from ..content_sources.fact_script_writer import write_script_from_facts
                from ..content_sources import lint_script

                try:
                    rewrite_result = write_script_from_facts(
                        facts,
                        language=plan.language,
                        target_blocks=plan.target_blocks,
                        style=plan.style or None,
                        extra_constraints=[f"Quality reviewer feedback: {feedback}"],
                    )
                    rewrite_script = apply_citation_cleanup(
                        rewrite_result["script"],
                        facts,
                        all_warnings,
                    )
                    rewrite_lint = lint_script(rewrite_script)

                    # Keep whichever is better
                    if rewrite_lint["score"] >= script_result.lint_score:
                        rewrite_script["image_mode"] = script_result.image_display
                        script_result = script_result.model_copy(update={
                            "script": rewrite_script,
                            "lint_score": rewrite_lint["score"],
                            "lint_status": rewrite_lint["status"],
                        })
                        self.state.script = rewrite_script
                        self._emit({"phase": "quality_gate", "message": "Rewrite accepted."})
                    else:
                        all_warnings.append("Quality gate rewrite scored lower — keeping original.")
                except Exception as exc:
                    all_warnings.append(f"Quality gate rewrite failed: {exc}")
            else:
                all_warnings.append(
                    f"Quality gate failed (det={gate.det_score}/50 llm={gate.llm_score}/50) but no feedback provided."
                )

        # Ensure phrase windows are present (quality gate rewrite bypasses script_agent's window pass)
        _blocks = script_result.script.get("blocks", [])
        if _blocks and _blocks[0].get("text") and not _blocks[0].get("windows"):
            try:
                from ..images.phrase_windows import split_into_windows
                _blocks[0]["windows"] = split_into_windows(_blocks[0]["text"], plan.topic)
            except Exception:
                pass

        # Save script after quality gate
        _save_artifact(run_dir, "script.json", script_result.script)

        # === PHASE 5: IMAGE ===
        self.state.phase = AgentPhase.IMAGE
        image_result = run_images(script_result.script, plan, emit=self._emit)
        all_warnings.extend(image_result.warnings)
        self.state.image_map = image_result.image_map

        # Save final script with images attached
        _save_artifact(run_dir, "script_final.json", image_result.script)

        # Inject subtitle preset from user config into script so manager picks it up
        if user_config and user_config.subtitle_preset:
            image_result.script["subtitle_preset"] = user_config.subtitle_preset

        # === PHASE 6: EDITOR ===
        self.state.phase = AgentPhase.EDITOR
        from ..agent_config import load_agent_settings
        _settings = load_agent_settings()
        _lab_cfg  = _settings.get("lab_editor", {}) or {}
        _editor_mode = _lab_cfg.get("editor_mode", "classic")
        if _editor_mode == "lab":
            editor_result = run_editor_lab(image_result.script, output_name, emit=self._emit)
        else:
            editor_result = run_editor(image_result.script, output_name, emit=self._emit)
        self.state.video_path = editor_result.get("video_path")
        self.state.audio_path = editor_result.get("audio_path")

        # === DONE ===
        self.state.phase = AgentPhase.DONE
        self._emit({"phase": "done", "message": "Video generation complete!"})
        self.state.warnings = all_warnings

        return AgentResult(
            success=True,
            video_path=self.state.video_path,
            audio_path=self.state.audio_path,
            script=script_result.script,
            lint_score=script_result.lint_score,
            sources_used=sources_used,
            images_matched=len(image_result.image_map),
            warnings=all_warnings,
            metadata={
                "topic": plan.topic,
                "language": plan.language,
                "style": plan.style,
                "topic_category": plan.topic_category,
                "image_display": script_result.image_display,
                "target_blocks": plan.target_blocks,
                "actual_blocks": len(script_result.script.get("blocks", [])),
                "facts_available": len(facts),
                "lint_status": script_result.lint_status,
                "writer_meta": script_result.writer_meta,
                "quality_gate": {
                    "passed": gate.passed,
                    "det_score": gate.det_score,
                    "llm_score": gate.llm_score,
                    "combined_score": gate.combined_score,
                    "det_issues": gate.det_issues,
                    "llm_feedback": gate.llm_feedback,
                    "skipped": gate.skipped,
                },
            },
        )


def _sanitize_filename(text: str) -> str:
    """Convert text to a safe filename token."""
    clean = re.sub(r'[^\w\s-]', '', text.lower().strip())
    clean = re.sub(r'[\s-]+', '_', clean)
    return clean[:60] or "agent_video"


def _save_artifact(run_dir: Path, filename: str, data: Any) -> None:
    """Save a JSON artifact to the run tracking folder. Never raises."""
    try:
        path = run_dir / filename
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Saved artifact: %s", path)
    except Exception as exc:
        logger.warning("Failed to save artifact %s: %s", filename, exc)
