"""Shared data contracts used across agent modules.

These dataclasses are the ONLY canonical representation of research / script
artifacts during pipeline execution. Do not add Pydantic here — the goal is
zero import-time dependencies so any module can import freely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _safe_hook(text: str, limit: int = 120) -> str:
    """Truncate at the last word boundary before `limit`. Never mid-word cuts."""
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    space = truncated.rfind(" ")
    return truncated[:space] if space > 0 else truncated


@dataclass
class ChunkRecord:
    """A single retrievable chunk, enriched with parent context."""
    chunk_id: str
    text: str
    section_heading: str = ""
    preceding_context: str = ""
    following_context: str = ""
    source_url: str = ""
    page_title: str = ""
    authority_tier: int = 4
    embedding: list[float] | None = None


@dataclass
class SourceRecord:
    """A fetched page, stored as the parent of its chunks."""
    url: str
    domain: str
    authority_tier: int
    title: str
    fetched_at: float
    raw_text: str
    chunks: list[ChunkRecord] = field(default_factory=list)


@dataclass
class GroundedFact:
    """A fact extracted from research and verified against its source."""
    fact_id: str
    claim: str
    verbatim_evidence: str
    source_url: str
    source_domain: str
    authority_tier: int
    extraction_confidence: float
    final_score: float
    grounded: bool = False
    topic_match: bool = False
    reason_tags: list[str] = field(default_factory=list)
    angle_served: str = ""

    def to_legacy_dict(self) -> dict[str, Any]:
        """Legacy shape used by script_agent until script_agent is rewritten."""
        return {
            "fact_text": self.claim,
            "source": "web",
            "score": self.final_score,
            "hook_text": _safe_hook(self.claim.split(". ", 1)[0]),
            "source_url": self.source_url,
            "reason_tags": self.reason_tags + ["grounded"] if self.grounded else self.reason_tags,
            "fact_id": self.fact_id,
            "verbatim_evidence": self.verbatim_evidence,
            "authority_tier": self.authority_tier,
            "angle_served": self.angle_served,
        }


@dataclass
class CoverageReport:
    """Diagnostic snapshot of what the research agent accomplished."""
    total_urls_fetched: int = 0
    tier_1_count: int = 0
    tier_2_count: int = 0
    tier_3_count: int = 0
    tier_4_count: int = 0
    chunks_indexed: int = 0
    chunks_retrieved: int = 0
    facts_extracted: int = 0
    facts_grounded: int = 0
    facts_failed_grounding: int = 0
    facts_failed_topic_match: int = 0
    min_fact_threshold_met: bool = False
    min_tier_diversity_met: bool = False

    def summary(self) -> str:
        return (
            f"URLs={self.total_urls_fetched} (T1={self.tier_1_count} T2={self.tier_2_count} "
            f"T3={self.tier_3_count} T4={self.tier_4_count}) "
            f"chunks={self.chunks_indexed}/{self.chunks_retrieved} "
            f"facts={self.facts_grounded}/{self.facts_extracted} "
            f"(failed_grounding={self.facts_failed_grounding}, failed_topic={self.facts_failed_topic_match})"
        )


@dataclass
class ResearchResult:
    """Output of the research agent."""
    facts: list[GroundedFact]
    source_manifest: list[SourceRecord]
    coverage: CoverageReport
    warnings: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0
    cache_hit: bool = False

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "facts": [f.to_legacy_dict() for f in self.facts],
            "sources_used": list({s.domain for s in self.source_manifest}),
            "search_context": "grounded-v2",
            "warnings": self.warnings,
            "coverage": self.coverage.summary(),
        }


@dataclass
class Citation:
    """Link between one script sentence and one research fact."""
    fact_id: str
    block_idx: int
    sentence_idx: int


@dataclass
class CitationMap:
    citations: list[Citation] = field(default_factory=list)
    unused_fact_ids: list[str] = field(default_factory=list)
    uncited_sentences: list[tuple[int, int]] = field(default_factory=list)

    @property
    def citation_rate(self) -> float:
        total = len(self.citations) + len(self.uncited_sentences)
        if total == 0:
            return 0.0
        return len(self.citations) / total
