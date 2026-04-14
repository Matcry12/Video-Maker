"""Data models for Topic Pool + Fact Bank persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


FactStatus = Literal["unused", "used", "archived"]
IngestStatus = Literal["ok", "error"]


class ScoreBreakdown(BaseModel):
    novelty: float = 0.0
    relevance: float = 0.0
    hook_strength: float = 0.0
    clarity: float = 0.0
    confidence: float = 0.0


class TopicBundle(BaseModel):
    topic_id: str
    topic_query: str
    language: str = "en-US"
    source: str = "source_1"
    wiki_title: str = ""
    wiki_url: str = ""
    summary_text: str = ""
    extended_text: str = ""
    fetched_at: str = Field(default_factory=lambda: utc_now_iso())
    ingest_status: IngestStatus = "ok"
    warnings: list[str] = Field(default_factory=list)


class FactCard(BaseModel):
    fact_id: str
    topic_id: str
    topic_label: str
    fact_text: str
    hook_text: str = ""
    evidence_text: str = ""
    source_url: str = ""
    language: str = "en-US"
    score: float = 0.0
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    status: FactStatus = "unused"
    created_at: str = Field(default_factory=lambda: utc_now_iso())
    used_at: Optional[str] = None
    used_in_videos: list[str] = Field(default_factory=list)


class BankIndex(BaseModel):
    version: int = 1
    topic_count: int = 0
    fact_count: int = 0
    unused_count: int = 0
    updated_at: str = Field(default_factory=lambda: utc_now_iso())


class IngestTopicResult(BaseModel):
    topic_id: str
    topic_query: str
    ingest_status: IngestStatus
    warning_count: int = 0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
