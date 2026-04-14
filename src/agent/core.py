"""Video generation agent — autonomous pipeline orchestrator."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .editor_agent import run_editor
from .image_agent import run_images
from .models import AgentConfig, AgentPhase, AgentPlan, AgentResult, AgentState
from .plan_agent import plan_from_prompt
from .quality_gate import run_quality_gate
from .research_agent import run_research
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
            "topic": plan.topic,
            "language": plan.language,
            "style": plan.style,
            "topic_category": plan.topic_category,
            "target_blocks": plan.target_blocks,
            "image_display": plan.image_display,
            "search_queries": plan.search_queries,
            "timestamp": run_ts,
        })

        # === PHASE 2: RESEARCH ===
        self.state.phase = AgentPhase.RESEARCH
        self._emit({"phase": "research", "message": f"Researching '{plan.topic}'..."})

        research_result = run_research(
            topic=plan.topic,
            search_queries=plan.search_queries,
            language=plan.language,
            emit=self._emit,
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
        })

        # === PHASE 3: SCRIPT ===
        self.state.phase = AgentPhase.SCRIPT
        script_result = run_script(facts, plan, emit=self._emit)
        all_warnings.extend(script_result.warnings)
        self.state.script = script_result.script

        # === PHASE 4: QUALITY GATE ===
        self.state.phase = AgentPhase.QUALITY_GATE
        self._emit({"phase": "quality_gate", "message": "Evaluating script quality..."})

        gate_result = run_quality_gate(script_result.script, plan.topic)

        if not gate_result["passed"] and not gate_result.get("skipped", False):
            # Fail → one targeted rewrite with feedback
            feedback = gate_result.get("feedback", "")
            if feedback:
                self._emit({"phase": "quality_gate", "message": "Script needs improvement, rewriting..."})
                all_warnings.append(
                    f"Quality gate failed (yes={gate_result['yes_count']}/3). "
                    f"Rewriting with feedback: {feedback[:100]}"
                )

                # Re-run script agent with quality gate feedback as extra constraint
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
                    rewrite_script = rewrite_result["script"]
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
                    f"Quality gate failed (yes={gate_result['yes_count']}/3) but no feedback provided."
                )

        # Save script after quality gate
        _save_artifact(run_dir, "script.json", script_result.script)

        # === PHASE 5: IMAGE ===
        self.state.phase = AgentPhase.IMAGE
        image_result = run_images(script_result.script, plan, emit=self._emit)
        all_warnings.extend(image_result.warnings)
        self.state.image_map = image_result.image_map

        # Save final script with images attached
        _save_artifact(run_dir, "script_final.json", image_result.script)

        # === PHASE 6: EDITOR ===
        self.state.phase = AgentPhase.EDITOR
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
                    "passed": gate_result["passed"],
                    "yes_count": gate_result["yes_count"],
                    "skipped": gate_result.get("skipped", False),
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
