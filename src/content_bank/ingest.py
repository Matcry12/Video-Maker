"""Wikipedia topic ingestion helpers for the content bank."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .extractor import extract_fact_cards
from .scoring import apply_scores, dedupe_fact_cards
from .models import IngestTopicResult, TopicBundle, utc_now_iso
from .store import ContentBankStore
from ..content_sources.wikipedia_source import fetch_page_extract, fetch_summary, search_page

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOPICS = 20


@dataclass
class IngestRunResult:
    created_topics: int = 0
    updated_topics: int = 0
    accepted_topics: int = 0
    skipped_topics: int = 0
    created_facts: int = 0
    updated_facts: int = 0
    extracted_facts: int = 0
    saved_facts: int = 0
    deduped_facts: int = 0
    bundles: list[TopicBundle] = field(default_factory=list)
    extraction_meta: list[dict] = field(default_factory=list)
    items: list[IngestTopicResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def normalize_topic_list(topics: list[str], max_topics: int = DEFAULT_MAX_TOPICS) -> tuple[list[str], list[str]]:
    """Normalize, dedupe, and trim topic list; returns (accepted, skipped)."""
    if max_topics <= 0:
        max_topics = DEFAULT_MAX_TOPICS

    accepted: list[str] = []
    skipped: list[str] = []
    seen = set()

    for raw in topics or []:
        topic = " ".join(str(raw or "").strip().split())
        if not topic:
            continue

        dedupe_key = topic.casefold()
        if dedupe_key in seen:
            skipped.append(topic)
            continue

        if len(accepted) >= max_topics:
            skipped.append(topic)
            continue

        seen.add(dedupe_key)
        accepted.append(topic)

    return accepted, skipped


def ingest_topics_from_wikipedia(
    store: ContentBankStore,
    topics: list[str],
    *,
    language: str = "en-US",
    max_topics: int = DEFAULT_MAX_TOPICS,
    source: str = "source_1",
    extract_facts: bool = True,
    facts_per_topic_target: int = 8,
    fact_prompt_override: str | None = None,
    extract_timeout_sec: float = 35.0,
    near_dup_threshold: float = 0.88,
) -> IngestRunResult:
    """Fetch topic bundles from Wikipedia and persist them into content bank."""
    accepted, skipped = normalize_topic_list(topics, max_topics=max_topics)
    run = IngestRunResult(accepted_topics=len(accepted), skipped_topics=len(skipped))

    if skipped:
        run.warnings.append(
            f"Skipped {len(skipped)} duplicate/overflow topics after normalization."
        )

    bundles: list[TopicBundle] = []

    for topic in accepted:
        topic_warnings: list[str] = []
        topic_id = store.compute_topic_id(topic_query=topic, language=language, source=source)

        try:
            search = search_page(topic, language)
            selected = search.get("selected")
            if not selected:
                topic_warnings.append("No matching Wikipedia page found.")
                bundles.append(
                    TopicBundle(
                        topic_id=topic_id,
                        topic_query=topic,
                        language=language,
                        source=source,
                        ingest_status="error",
                        warnings=topic_warnings,
                        fetched_at=utc_now_iso(),
                    )
                )
                run.items.append(
                    IngestTopicResult(
                        topic_id=topic_id,
                        topic_query=topic,
                        ingest_status="error",
                        warning_count=len(topic_warnings),
                    )
                )
                continue

            title = str(selected.get("title") or "").strip()
            summary_payload = fetch_summary(title, language)
            summary_text = str(summary_payload.get("extract") or "").strip()
            if len(summary_text) < 140:
                topic_warnings.append("Summary is short; extended extract fetched.")

            extended_text = fetch_page_extract(title, language)
            if not extended_text:
                topic_warnings.append("Extended extract empty; using summary only.")

            bundles.append(
                TopicBundle(
                    topic_id=topic_id,
                    topic_query=topic,
                    language=language,
                    source=source,
                    wiki_title=summary_payload.get("title") or title,
                    wiki_url=summary_payload.get("canonical_url") or "",
                    summary_text=summary_text,
                    extended_text=extended_text,
                    ingest_status="ok",
                    warnings=topic_warnings,
                    fetched_at=utc_now_iso(),
                )
            )
            run.items.append(
                IngestTopicResult(
                    topic_id=topic_id,
                    topic_query=topic,
                    ingest_status="ok",
                    warning_count=len(topic_warnings),
                )
            )
        except Exception as exc:
            logger.warning("Topic ingest failed for '%s': %s", topic, exc)
            topic_warnings.append(str(exc))
            bundles.append(
                TopicBundle(
                    topic_id=topic_id,
                    topic_query=topic,
                    language=language,
                    source=source,
                    ingest_status="error",
                    warnings=topic_warnings,
                    fetched_at=utc_now_iso(),
                )
            )
            run.items.append(
                IngestTopicResult(
                    topic_id=topic_id,
                    topic_query=topic,
                    ingest_status="error",
                    warning_count=len(topic_warnings),
                )
            )

    created, updated = store.upsert_topics(bundles)
    run.created_topics = created
    run.updated_topics = updated
    run.bundles = bundles

    if extract_facts:
        existing_facts = store.load_facts()
        extracted_cards = []
        extraction_meta = []

        for bundle in bundles:
            if bundle.ingest_status != "ok":
                continue

            cards, meta, extraction_warnings = extract_fact_cards(
                bundle,
                facts_per_topic_target=facts_per_topic_target,
                prompt_override=fact_prompt_override,
                timeout_sec=extract_timeout_sec,
            )
            run.extracted_facts += len(cards)
            extracted_cards.extend(cards)
            extraction_meta.append(
                {
                    "topic_id": bundle.topic_id,
                    "topic_query": bundle.topic_query,
                    "facts_extracted": len(cards),
                    "meta": meta,
                    "warnings": extraction_warnings,
                }
            )
            for warning in extraction_warnings:
                run.warnings.append(f"{bundle.topic_query}: {warning}")

        run.extraction_meta = extraction_meta

        if extracted_cards:
            apply_scores(extracted_cards)
            deduped_cards, removed_count = dedupe_fact_cards(
                extracted_cards,
                existing=existing_facts,
                near_dup_threshold=near_dup_threshold,
            )
            run.deduped_facts = removed_count
            run.saved_facts = len(deduped_cards)

            if deduped_cards:
                created_facts, updated_facts = store.upsert_facts(deduped_cards)
                run.created_facts = created_facts
                run.updated_facts = updated_facts
    return run
