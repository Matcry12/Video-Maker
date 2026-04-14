"""Persistent storage layer for Topic Pool + Fact Bank."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Optional

from .models import BankIndex, FactCard, TopicBundle, utc_now_iso

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BANK_DIR = PROJECT_ROOT / "content_bank"


def _model_dump(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


class ContentBankStore:
    """Local JSON-backed persistence with atomic writes."""

    def __init__(self, bank_dir: Path | None = None):
        self.bank_dir = Path(bank_dir or DEFAULT_BANK_DIR)
        self.topics_path = self.bank_dir / "topics.json"
        self.facts_path = self.bank_dir / "facts.json"
        self.index_path = self.bank_dir / "index.json"
        self._lock = threading.Lock()
        self._ensure_initialized()

    def _ensure_initialized(self):
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        if not self.topics_path.exists():
            self._atomic_write_json(self.topics_path, [])
        if not self.facts_path.exists():
            self._atomic_write_json(self.facts_path, [])
        if not self.index_path.exists():
            self._atomic_write_json(self.index_path, _model_dump(BankIndex()))

    def load_topics(self) -> list[TopicBundle]:
        raw = self._read_json(self.topics_path, default=[])
        topics: list[TopicBundle] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                topics.append(TopicBundle(**item))
            except Exception as exc:
                logger.warning("Skipping invalid topic entry: %s", exc)
        return topics

    def load_facts(self) -> list[FactCard]:
        raw = self._read_json(self.facts_path, default=[])
        facts: list[FactCard] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                facts.append(FactCard(**item))
            except Exception as exc:
                logger.warning("Skipping invalid fact entry: %s", exc)
        return facts

    def load_index(self) -> BankIndex:
        raw = self._read_json(self.index_path, default={})
        if not isinstance(raw, dict):
            return BankIndex()
        try:
            return BankIndex(**raw)
        except Exception:
            return BankIndex()

    def refresh_index(
        self,
        *,
        topics: Optional[list[TopicBundle]] = None,
        facts: Optional[list[FactCard]] = None,
    ) -> BankIndex:
        topic_items = topics if topics is not None else self.load_topics()
        fact_items = facts if facts is not None else self.load_facts()
        unused_count = sum(1 for fact in fact_items if fact.status == "unused")

        index = BankIndex(
            version=1,
            topic_count=len(topic_items),
            fact_count=len(fact_items),
            unused_count=unused_count,
            updated_at=utc_now_iso(),
        )
        self._atomic_write_json(self.index_path, _model_dump(index))
        return index

    def upsert_topics(self, topics: list[TopicBundle]) -> tuple[int, int]:
        """Insert or update topics by topic_id. Returns (created, updated)."""
        if not topics:
            return 0, 0

        with self._lock:
            existing = {item.topic_id: item for item in self.load_topics()}
            created = 0
            updated = 0
            for topic in topics:
                if topic.topic_id in existing:
                    updated += 1
                else:
                    created += 1
                existing[topic.topic_id] = topic

            ordered = sorted(existing.values(), key=lambda item: (item.topic_query.casefold(), item.topic_id))
            self._atomic_write_json(self.topics_path, [_model_dump(item) for item in ordered])
            self.refresh_index(topics=ordered)
            return created, updated

    def upsert_facts(self, facts: list[FactCard]) -> tuple[int, int]:
        """Insert or update facts by fact_id. Returns (created, updated)."""
        if not facts:
            return 0, 0

        with self._lock:
            existing = {item.fact_id: item for item in self.load_facts()}
            created = 0
            updated = 0
            for fact in facts:
                if fact.fact_id in existing:
                    updated += 1
                else:
                    created += 1
                existing[fact.fact_id] = fact

            ordered = sorted(existing.values(), key=lambda item: (-item.score, item.created_at, item.fact_id))
            self._atomic_write_json(self.facts_path, [_model_dump(item) for item in ordered])
            self.refresh_index(facts=ordered)
            return created, updated

    def list_facts(
        self,
        *,
        status: Optional[str] = None,
        topic_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[FactCard], int]:
        facts = self.load_facts()
        items = facts

        if status:
            status_norm = str(status).strip().lower()
            items = [item for item in items if item.status == status_norm]

        if topic_id:
            topic_norm = str(topic_id).strip()
            items = [item for item in items if item.topic_id == topic_norm]

        total = len(items)
        start = max(0, int(offset))
        end = start + max(1, min(int(limit), 500))
        return items[start:end], total

    def mark_facts_used(self, fact_ids: list[str], video_id: str | None = None) -> int:
        if not fact_ids:
            return 0

        wanted = {str(item).strip() for item in fact_ids if str(item).strip()}
        if not wanted:
            return 0

        with self._lock:
            facts = self.load_facts()
            changed = 0
            now = utc_now_iso()
            video = str(video_id or "").strip()
            for fact in facts:
                if fact.fact_id not in wanted:
                    continue
                if fact.status != "used":
                    fact.status = "used"
                    fact.used_at = now
                    changed += 1
                if video and video not in fact.used_in_videos:
                    fact.used_in_videos.append(video)

            if changed or video:
                self._atomic_write_json(self.facts_path, [_model_dump(item) for item in facts])
                self.refresh_index(facts=facts)
            return changed

    def mark_facts_archived(self, fact_ids: list[str]) -> int:
        if not fact_ids:
            return 0
        wanted = {str(item).strip() for item in fact_ids if str(item).strip()}
        if not wanted:
            return 0

        with self._lock:
            facts = self.load_facts()
            changed = 0
            for fact in facts:
                if fact.fact_id in wanted and fact.status != "archived":
                    fact.status = "archived"
                    changed += 1
            if changed:
                self._atomic_write_json(self.facts_path, [_model_dump(item) for item in facts])
                self.refresh_index(facts=facts)
            return changed


    def compute_topic_id(self, topic_query: str, language: str, source: str = "source_1") -> str:
        canonical = {
            "topic_query": str(topic_query or "").strip().casefold(),
            "language": str(language or "").strip().lower(),
            "source": str(source or "").strip().lower(),
            "version": 1,
        }
        raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _read_json(self, path: Path, *, default):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed reading JSON file %s: %s", path, exc)
            return default

    def _atomic_write_json(self, path: Path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)
