"""Agent data models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentPhase(str, Enum):
    PLAN = "plan"
    RESEARCH = "research"
    SCRIPT = "script"
    QUALITY_GATE = "quality_gate"
    IMAGE = "image"
    EDITOR = "editor"
    DONE = "done"
    FAILED = "failed"


class AgentConfig(BaseModel):
    """User-provided overrides. None = let agents decide."""
    image_display: Optional[str] = None      # "popup" | "background"
    target_blocks: Optional[int] = None      # 3-10
    style: Optional[str] = None              # "dramatic"|"educational"|"cinematic"|"energetic"
    voice: Optional[str] = None              # TTS voice name
    bgm_mood: Optional[str] = None           # "intense"|"calm"|"mystery"|"epic"|"emotional"
    skill_id: Optional[str] = None           # force a specific skill from skills/*.json
    subtitle_preset: Optional[str] = None    # "minimal"|"energetic"|"cinematic"


class AgentPlan(BaseModel):
    topic: str
    language: str = "en-US"
    style: str = ""
    topic_category: str = ""                 # "history"|"anime"|"science"|"biography"|"trending"|""
    content_type: str = ""                   # "lore"|"dark_secrets"|"theory"|"easter_eggs"|"comparison"|""
    mood: str = ""                           # "dark_mystery"|"epic"|"emotional"|"funny"|"shocking"|""
    hook_strategy: str = ""                  # "lead_with_number"|"lead_with_question"|"lead_with_statement"|"lead_with_contrast"|""
    target_blocks: int = 6
    search_queries: list[str] = Field(default_factory=list)
    entity_aliases: list[str] = Field(default_factory=list)
    domain_preferences: list[str] = Field(default_factory=list)
    user_prompt: str = ""          # verbatim user free-text
    must_cover: list[str] = Field(default_factory=list)     # angles extracted from user_prompt
    entity_cards: list[dict] = Field(default_factory=list)  # per-entity decomposition from plan LLM
    narrative_dynamic: str = ""    # contrast/transformation the user is asking about
    forbidden_entities: list[str] = Field(default_factory=list)
    image_display: str = "popup"
    bgm_mood: str = ""                       # "intense"|"calm"|"mystery"|"epic"|"emotional"|""
    voice: str = ""                          # TTS voice, empty = auto-select by language
    user_overrides: Optional[AgentConfig] = None


class CrawlResult(BaseModel):
    """Output of CrawlAgent."""
    facts: list[dict[str, Any]] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ScriptResult(BaseModel):
    """Output of ScriptAgent."""
    script: dict[str, Any] = Field(default_factory=dict)
    image_display: str = "popup"
    lint_score: int = 0
    lint_status: str = "pass"
    writer_meta: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ImageResult(BaseModel):
    """Output of ImageAgent."""
    script: dict[str, Any] = Field(default_factory=dict)
    image_map: dict[int, list[str]] = Field(default_factory=dict)
    total_images: int = 0
    warnings: list[str] = Field(default_factory=list)
    dropped_keywords: dict[int, list[str]] = Field(default_factory=dict)


class AgentState(BaseModel):
    phase: AgentPhase = AgentPhase.PLAN
    plan: Optional[AgentPlan] = None
    script: dict[str, Any] = Field(default_factory=dict)
    lint_result: dict[str, Any] = Field(default_factory=dict)
    image_map: dict[int, list[str]] = Field(default_factory=dict)
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    retry_count: int = 0


class AgentResult(BaseModel):
    success: bool
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    script: dict[str, Any] = Field(default_factory=dict)
    lint_score: int = 0
    sources_used: list[str] = Field(default_factory=list)
    images_matched: int = 0
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
