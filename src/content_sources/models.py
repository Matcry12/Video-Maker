"""Shared source-processing models for crawl extraction."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    doc_id: str
    title: str = ""
    url: str = ""
    text: str
    language: str = "en-US"
    position: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


class SourceUnit(BaseModel):
    unit_id: str
    source: str
    topic_query: str
    doc_id: str
    title: str = ""
    url: str = ""
    text: str
    language: str = "en-US"
    position: int = 0
    local_score: float = 0.0
    local_signals: dict[str, Any] = Field(default_factory=dict)


class SourceFetchResult(BaseModel):
    source: str
    topic_query: str
    language: str = "en-US"
    title: str = ""
    source_url: str = ""
    fetched_at: str = ""
    documents: list[SourceDocument] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
