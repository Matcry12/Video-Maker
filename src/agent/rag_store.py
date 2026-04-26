"""ChromaDB-backed semantic cache for research chunks.

Each chunk is stored with rich metadata so retrieval can:
  - weight results by source authority tier
  - return the window of neighbouring chunks as parent-doc context
  - filter by topic aliases
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SUPPRESS_LOGS = ["sentence_transformers", "chromadb", "httpx", "urllib3"]
for _name in _SUPPRESS_LOGS:
    logging.getLogger(_name).setLevel(logging.ERROR)

_STORE_DIR = Path(__file__).parent.parent.parent / ".omc" / "rag_cache"
_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_DEFAULT_TTL = 7 * 24 * 3600

_embedder = None
_chroma_client = None
_reranker = None

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(_RERANKER_MODEL)
        except Exception as exc:
            logger.debug("CrossEncoder unavailable: %s", exc)
            _reranker = False  # sentinel: don't retry
    return _reranker if _reranker is not False else None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(_EMBED_MODEL)
    return _embedder


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _STORE_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(_STORE_DIR))
    return _chroma_client


def _col_name(topic: str) -> str:
    h = hashlib.sha256(topic.lower().strip().encode()).hexdigest()[:12]
    return f"research-{h}"


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


class RagStore:
    """Per-topic vector store with tier-weighted hybrid retrieval."""

    def __init__(self, topic: str, ttl_secs: int = _DEFAULT_TTL):
        self.topic = topic
        self._name = _col_name(topic)
        self._ttl = ttl_secs

    # ------------------------------------------------------------------
    # Cache status
    # ------------------------------------------------------------------
    def count(self) -> int:
        try:
            col = _get_client().get_or_create_collection(self._name)
            return col.count()
        except Exception:
            return 0

    def is_cached(self) -> bool:
        """Return True iff the collection has chunks AND the newest chunk is
        younger than TTL."""
        try:
            col = _get_client().get_or_create_collection(self._name)
            if col.count() == 0:
                return False
            result = col.get(limit=1, include=["metadatas"])
            metas = result.get("metadatas") or []
            if not metas:
                return False
            cached_at = _safe_float(metas[0].get("cached_at", "0"), 0.0)
            return (time.time() - cached_at) < self._ttl
        except Exception as exc:
            logger.debug("is_cached failed: %s", exc)
            return False

    def clear(self) -> None:
        try:
            _get_client().delete_collection(self._name)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add_chunks(self, chunks: list[dict]) -> int:
        """Upsert a list of chunk dicts. Each dict must provide:
          - text (str, required)
          - source_url (str)
          - page_title (str)
          - authority_tier (int 1-4)
          - section_heading (str)
          - preceding_context (str)
          - following_context (str)
          - chunk_idx (int, position inside its parent page)
        Returns count stored.
        """
        if not chunks:
            return 0
        try:
            col = _get_client().get_or_create_collection(self._name)
            texts: list[str] = []
            metadatas: list[dict] = []
            ids: list[str] = []
            now_str = str(time.time())
            for i, c in enumerate(chunks):
                text = (c.get("text") or "").strip()
                if len(text) < 50:
                    continue
                url = str(c.get("source_url", ""))
                idx = int(c.get("chunk_idx", i))
                chunk_id = f"{self._name}:{hashlib.md5(url.encode()).hexdigest()[:10]}:{idx}"
                ids.append(chunk_id)
                texts.append(text)
                metadatas.append({
                    "source_url": url,
                    "page_title": str(c.get("page_title", "")),
                    "authority_tier": str(int(c.get("authority_tier", 4))),
                    "section_heading": str(c.get("section_heading", ""))[:200],
                    "preceding_context": str(c.get("preceding_context", ""))[:200],
                    "following_context": str(c.get("following_context", ""))[:200],
                    "chunk_idx": str(idx),
                    "cached_at": now_str,
                })
            if not texts:
                return 0
            embedder = _get_embedder()
            embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
            col.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            logger.info("RagStore[%s]: stored %d chunks", self.topic, len(texts))
            return len(texts)
        except Exception as exc:
            logger.warning("RagStore.add_chunks failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def retrieve(
        self,
        queries: list[str],
        top_k: int = 12,
        min_tier: int | None = None,
        rrf_k: int = 60,
        tier_weights: dict[int, float] | None = None,
    ) -> list[dict]:
        """Hybrid retrieval: vector search + BM25, merged via tier-weighted RRF.

        Vector search catches semantic matches; BM25 catches exact proper-noun
        matches (e.g. "Ashborn", "Mahoraga") that embeddings may miss.
        Both contribute `tier_weight / (rrf_k + rank)` to the final score.
        """
        try:
            embedder = _get_embedder()
            col = _get_client().get_or_create_collection(self._name)
            total = col.count()
            if total == 0:
                return []
            n_results = min(top_k * 3, total)
        except Exception:
            return []

        tw = tier_weights or {1: 1.5, 2: 1.0, 3: 0.4, 4: 0.25}
        rrf_scores: dict[str, float] = {}
        rrf_docs: dict[str, dict] = {}

        def _register(doc_id: str, doc: str, meta: dict, rank: int) -> None:
            tier = _safe_int(meta.get("authority_tier"), 4)
            if min_tier is not None and tier > min_tier:
                return
            weight = tw.get(tier, 0.25)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight / (rank + rrf_k)
            if doc_id not in rrf_docs:
                rrf_docs[doc_id] = {
                    "chunk_id": doc_id,
                    "text": doc,
                    "source_url": meta.get("source_url", ""),
                    "page_title": meta.get("page_title", ""),
                    "authority_tier": tier,
                    "section_heading": meta.get("section_heading", ""),
                    "preceding_context": meta.get("preceding_context", ""),
                    "following_context": meta.get("following_context", ""),
                    "chunk_idx": _safe_int(meta.get("chunk_idx"), 0),
                }

        # --- Vector search leg ---
        for query in (queries or [])[:6]:
            if not query:
                continue
            try:
                q_emb = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False).tolist()
                result = col.query(
                    query_embeddings=q_emb,
                    n_results=n_results,
                    include=["documents", "metadatas"],
                )
                docs = (result.get("documents") or [[]])[0]
                metas = (result.get("metadatas") or [[]])[0]
                ids = (result.get("ids") or [[]])[0]
                for rank, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas)):
                    _register(doc_id, doc, meta, rank)
            except Exception as exc:
                logger.debug("vector retrieve query %r failed: %s", query, exc)

        # --- BM25 search leg ---
        try:
            from rank_bm25 import BM25Okapi
            all_data = col.get(include=["documents", "metadatas"])
            all_ids = all_data.get("ids") or []
            all_docs = all_data.get("documents") or []
            all_metas = all_data.get("metadatas") or []
            if all_docs:
                tokenized = [d.lower().split() for d in all_docs]
                bm25 = BM25Okapi(tokenized)
                for query in (queries or [])[:6]:
                    if not query:
                        continue
                    q_tokens = query.lower().split()
                    scores = bm25.get_scores(q_tokens)
                    # Take top n_results by BM25 score
                    ranked_idxs = sorted(range(len(scores)), key=lambda i: -scores[i])[:n_results]
                    for rank, idx in enumerate(ranked_idxs):
                        if scores[idx] <= 0:
                            break
                        _register(all_ids[idx], all_docs[idx], all_metas[idx], rank)
        except Exception as exc:
            logger.debug("BM25 retrieval failed (non-fatal): %s", exc)

        # RRF initial ranking — take 2× top_k candidates for reranker
        candidates = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k * 2]
        out = []
        for doc_id, score in candidates:
            if doc_id in rrf_docs:
                d = dict(rrf_docs[doc_id])
                d["rrf_score"] = score
                out.append(d)

        # Cross-encoder reranking — re-scores (query, chunk) pairs more accurately
        reranker = _get_reranker()
        if reranker is not None and out and queries:
            try:
                # Use the first query as the reranking anchor (most specific)
                anchor = next((q for q in queries if q), "")
                if anchor:
                    pairs = [(anchor, d["text"]) for d in out]
                    ce_scores = reranker.predict(pairs, show_progress_bar=False)
                    for d, ce_score in zip(out, ce_scores):
                        d["ce_score"] = float(ce_score)
                    out.sort(key=lambda d: -d.get("ce_score", 0.0))
            except Exception as exc:
                logger.debug("Cross-encoder reranking failed (non-fatal): %s", exc)

        return out[:top_k]

    def get_parent_window(self, chunk_id: str, window: int = 2) -> list[dict]:
        """Return the chunk `chunk_id` plus `window` chunks before and after
        within the same parent page. Used to give extractor full context."""
        try:
            col = _get_client().get_or_create_collection(self._name)
            # Parse chunk_id: "{collection}:{url_hash}:{idx}"
            parts = chunk_id.split(":")
            if len(parts) < 3:
                return []
            url_hash = parts[-2]
            idx = int(parts[-1])
            # Fetch any chunk with matching url_hash prefix by scanning IDs
            # (small collections: this is OK)
            result = col.get(include=["documents", "metadatas"])
            ids = result.get("ids") or []
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []
            sibling = []
            for did, doc, meta in zip(ids, docs, metas):
                dparts = did.split(":")
                if len(dparts) < 3 or dparts[-2] != url_hash:
                    continue
                didx = _safe_int(dparts[-1], 0)
                if abs(didx - idx) <= window:
                    sibling.append((didx, {
                        "chunk_id": did,
                        "text": doc,
                        "chunk_idx": didx,
                        "source_url": meta.get("source_url", ""),
                        "page_title": meta.get("page_title", ""),
                        "authority_tier": _safe_int(meta.get("authority_tier"), 4),
                        "section_heading": meta.get("section_heading", ""),
                    }))
            sibling.sort(key=lambda x: x[0])
            return [s[1] for s in sibling]
        except Exception as exc:
            logger.debug("get_parent_window failed: %s", exc)
            return []


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return default
