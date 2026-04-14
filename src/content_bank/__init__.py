"""Topic Pool + Fact Bank package."""

from .extractor import extract_fact_cards
from .ingest import DEFAULT_MAX_TOPICS, IngestRunResult, ingest_topics_from_wikipedia, normalize_topic_list
from .models import BankIndex, FactCard, IngestTopicResult, ScoreBreakdown, TopicBundle
from .scoring import apply_scores, dedupe_fact_cards
from .selector import SUPPORTED_SELECTION_MODES, build_script_from_facts, select_facts
from .store import ContentBankStore

__all__ = [
    "BankIndex",
    "DEFAULT_MAX_TOPICS",
    "FactCard",
    "IngestRunResult",
    "IngestTopicResult",
    "ScoreBreakdown",
    "TopicBundle",
    "ContentBankStore",
    "SUPPORTED_SELECTION_MODES",
    "apply_scores",
    "build_script_from_facts",
    "dedupe_fact_cards",
    "extract_fact_cards",
    "ingest_topics_from_wikipedia",
    "normalize_topic_list",
    "select_facts",
]
