"""Lightweight in-memory RAG indices for research retrieval.

BM25 (keyword) + Dense (semantic) hybrid retrieval.
Both indices are ephemeral — built per research run, cleared after retrieval.
"""

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9\u00C0-\u024F\u1E00-\u1EFF]+", re.UNICODE)


class RAGIndex:
    """In-memory BM25 index over text chunks."""

    def __init__(self):
        self._documents: list[dict[str, Any]] = []
        self._tokenized: list[list[str]] = []
        self._index = None  # BM25Okapi, built lazily

    def add(self, doc: dict[str, Any]) -> None:
        """Add a document dict (must have 'text' key)."""
        text = str(doc.get("text", ""))
        tokens = _tokenize(text)
        if not tokens:
            return
        self._documents.append(doc)
        self._tokenized.append(tokens)
        self._index = None  # invalidate

    def add_many(self, docs: list[dict[str, Any]]) -> None:
        for doc in docs:
            self.add(doc)

    def build(self) -> None:
        """Build/rebuild the BM25 index. Called automatically on first query."""
        if not self._tokenized:
            return
        from rank_bm25 import BM25Okapi
        self._index = BM25Okapi(self._tokenized)

    def query(self, query_text: str, top_k: int = 15) -> list[dict[str, Any]]:
        """Retrieve top_k documents by BM25 relevance."""
        if not self._documents:
            return []
        if self._index is None:
            self.build()
        tokens = _tokenize(query_text)
        if not tokens:
            return self._documents[:top_k]
        scores = self._index.get_scores(tokens)
        ranked = sorted(
            zip(scores, range(len(self._documents)), self._documents),
            key=lambda t: (-t[0], t[1]),
        )
        return [doc for _score, _idx, doc in ranked[:top_k]]

    def query_with_scores(self, query_text: str, top_k: int = 15) -> list[tuple[dict[str, Any], float]]:
        """Retrieve top_k documents with their BM25 scores."""
        if not self._documents:
            return []
        if self._index is None:
            self.build()
        tokens = _tokenize(query_text)
        if not tokens:
            return [(doc, 0.0) for doc in self._documents[:top_k]]
        scores = self._index.get_scores(tokens)
        ranked = sorted(
            zip(scores, range(len(self._documents)), self._documents),
            key=lambda t: (-t[0], t[1]),
        )
        return [(doc, score) for score, _idx, doc in ranked[:top_k]]

    def size(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        """Free all memory."""
        self._documents.clear()
        self._tokenized.clear()
        self._index = None


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) > 1]


# ---------------------------------------------------------------------------
# Dense embedding index (fastembed / ONNX)
# ---------------------------------------------------------------------------

_DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_dense_model_cache = None  # Module-level cache — loaded once per process


def _get_dense_model():
    """Load and cache the fastembed model (~130MB, ONNX, CPU-only)."""
    global _dense_model_cache
    if _dense_model_cache is not None:
        return _dense_model_cache
    try:
        from fastembed import TextEmbedding
        logger.info("Loading dense embedding model '%s'...", _DENSE_MODEL_NAME)
        _dense_model_cache = TextEmbedding(model_name=_DENSE_MODEL_NAME)
        logger.info("Dense model loaded.")
        return _dense_model_cache
    except Exception as exc:
        logger.warning("Failed to load dense model: %s", exc)
        return None


class DenseIndex:
    """In-memory dense vector index using fastembed ONNX embeddings."""

    def __init__(self):
        self._documents: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None  # (N, dim)
        self._model = None

    def add_many(self, docs: list[dict[str, Any]]) -> None:
        """Add documents and compute embeddings for all texts."""
        texts = []
        valid_docs = []
        for doc in docs:
            text = str(doc.get("text", "")).strip()
            if text:
                texts.append(text[:512])  # cap for embedding model context
                valid_docs.append(doc)

        if not texts:
            return

        self._model = _get_dense_model()
        if self._model is None:
            self._documents = valid_docs
            return

        # Encode all texts in one batch
        embeddings = list(self._model.embed(texts))
        self._embeddings = np.array(embeddings, dtype=np.float32)
        self._documents = valid_docs

    def query(self, query_text: str, top_k: int = 15) -> list[tuple[dict[str, Any], float]]:
        """Retrieve top_k documents by cosine similarity."""
        if not self._documents:
            return []
        if self._model is None or self._embeddings is None:
            return [(doc, 0.0) for doc in self._documents[:top_k]]

        # Encode query
        q_emb = list(self._model.embed([query_text[:512]]))[0]
        q_emb = np.array(q_emb, dtype=np.float32)

        # Cosine similarity (embeddings are already normalized by fastembed)
        scores = self._embeddings @ q_emb

        ranked = sorted(
            zip(scores, range(len(self._documents)), self._documents),
            key=lambda t: (-t[0], t[1]),
        )
        return [(doc, float(score)) for score, _idx, doc in ranked[:top_k]]

    def size(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        """Free embeddings memory. Model stays cached at module level."""
        self._documents.clear()
        self._embeddings = None
        self._model = None


# ---------------------------------------------------------------------------
# Hybrid retrieval: BM25 + Dense with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    bm25_index: RAGIndex,
    dense_index: DenseIndex,
    query: str,
    top_k: int = 20,
    rrf_k: int = 60,
) -> list[tuple[dict[str, Any], float]]:
    """Combine BM25 and dense retrieval using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) across both result lists.
    Returns top_k documents sorted by fused score.
    """
    # Get ranked results from both
    bm25_results = bm25_index.query_with_scores(query, top_k=top_k)
    dense_results = dense_index.query(query, top_k=top_k)

    # Build RRF scores keyed by document identity
    # Use text[:100] as doc key since unit_id may not be unique
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, (doc, _score) in enumerate(bm25_results):
        key = str(doc.get("unit_id", doc.get("text", "")[:100]))
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key] = doc

    for rank, (doc, _score) in enumerate(dense_results):
        key = str(doc.get("unit_id", doc.get("text", "")[:100]))
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key] = doc

    # Sort by fused score
    ranked = sorted(rrf_scores.items(), key=lambda t: -t[1])
    return [(doc_map[key], score) for key, score in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Topic-entity relevance filter
# ---------------------------------------------------------------------------

_TOPIC_SPLIT_WORDS = {
    "'s", "in", "of", "about", "from", "with", "for", "the",
    "trong", "của", "về", "và", "bí", "mật", "sự", "thật",
    "đen", "tối", "hay", "nhất",
}


def extract_topic_entities(topic: str) -> list[str]:
    """Extract searchable entity names from a topic string.

    Splits on prepositions and particles, keeps meaningful tokens.
    Returns lowercased entity strings for fuzzy matching.
    """
    # Lowercase
    lower = topic.lower().strip()

    # Remove common prefixes
    for prefix in ["sự thật đen tối về ", "sự thật về ", "bí mật về ",
                    "dark secrets of ", "hidden details of ", "facts about "]:
        if lower.startswith(prefix):
            lower = lower[len(prefix):]
            break

    # Split into tokens, filter out stop words
    tokens = re.split(r"[\s,.']+", lower)
    tokens = [t for t in tokens if t and len(t) >= 2 and t not in _TOPIC_SPLIT_WORDS]

    entities: list[str] = []

    # Add individual meaningful tokens (proper nouns, names)
    for t in tokens:
        if len(t) >= 3:
            entities.append(t)

    # For multi-word names, also add the LAST token as standalone
    # (catches "Kirara" from "Hoshi Kirara", "Bernard" from "Charles Bernard")
    if len(tokens) >= 2:
        entities.append(tokens[-1])
        entities.append(tokens[0])

    # Add consecutive pairs (catches "charles bernard", "re zero", etc.)
    for i in range(len(tokens) - 1):
        pair = f"{tokens[i]} {tokens[i+1]}"
        if len(pair) >= 5:
            entities.append(pair)

    # Add the cleaned full topic
    cleaned_full = " ".join(tokens)
    if cleaned_full and len(cleaned_full) >= 3:
        entities.append(cleaned_full)

    # Common aliases: "Re:Zero" → "rezero", "JJK" → "jujutsu kaisen"
    _ALIASES = {
        "re:zero": ["rezero", "re zero"],
        "jjk": ["jujutsu kaisen"],
        "jujutsu kaisen": ["jjk"],
        "one piece": ["onepiece"],
        "dragon ball": ["dragonball", "dbz"],
        "aot": ["attack on titan", "shingeki"],
        "bnha": ["my hero academia", "boku no hero"],
    }
    for entity in list(entities):
        for key, aliases in _ALIASES.items():
            if key in entity:
                entities.extend(aliases)

    return list(set(entities))


def filter_by_topic_relevance(
    items: list[tuple[dict, float]],
    topic: str,
    penalty: float = 0.1,
    min_relevant: int = 4,
) -> list[tuple[dict, float]]:
    """Penalize items that don't mention the topic entity.

    Items mentioning the entity keep their score.
    Items NOT mentioning it get score * penalty.
    If fewer than min_relevant items mention the entity, keep all scores unchanged.
    """
    entities = extract_topic_entities(topic)
    if not entities:
        return items

    def _mentions_entity(text: str) -> bool:
        text_lower = text.lower()
        return any(e in text_lower for e in entities)

    # Count how many mention the entity
    relevant_count = sum(
        1 for doc, _score in items
        if _mentions_entity(doc.get("text", ""))
    )

    # If too few relevant, don't penalize — better to keep all than have 2 facts
    if relevant_count < min_relevant:
        logger.info(
            "Topic filter: only %d/%d mention entity — skipping penalty",
            relevant_count, len(items),
        )
        return items

    # Apply penalty
    adjusted = []
    for doc, score in items:
        if _mentions_entity(doc.get("text", "")):
            adjusted.append((doc, score))
        else:
            adjusted.append((doc, score * penalty))

    # Re-sort by adjusted score
    adjusted.sort(key=lambda t: -t[1])

    logger.info(
        "Topic filter: %d/%d relevant, %d penalized (entities=%s)",
        relevant_count, len(items) - relevant_count,
        len(items) - relevant_count, entities[:5],
    )
    return adjusted
