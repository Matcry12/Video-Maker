"""Trend brainstorm agent — combines trend signals with LLM brainstorming."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
TREND_LOG = PROJECT_ROOT / ".omc" / "trend_log.json"
BRAINSTORM_CACHE = PROJECT_ROOT / ".omc" / "brainstorm_sem_cache.json"
_PROMPT_DIR = PROJECT_ROOT / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


_JSON_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_PUNCT_RE = re.compile(r"[^\w\s]")

_SEVEN_DAYS_SECS = 7 * 24 * 3600
_CACHE_TTL_SECS = 24 * 3600

_ANIME_SKILL_GUIDE = """\
- New episode drop, trivia, surprising details → "did_you_know"
- "vs", power ranking, who wins → "comparison"
- Theory, plot twist, ending explained → "theory"
- Dark past, hidden truth, secret, buried lore → "dark_secrets"
- Series origin, debut, founding story → "story_time"
- World-building, mythology, magic system → "lore_deep_dive"
- Background details, hidden references, easter eggs → "easter_eggs"
- Funny moments, absurd premise, comedy → "comedy"
- Top 5 / Top 10 countdown → "top_list"
- Timeline of events, chronology → "timeline"
- Myth busting, misconception corrected → "debunked"
- Character/concept explained → "explained"
- What happened after the ending → "what_happened_to"
"""

_GAMING_SKILL_GUIDE = """\
- Patch, update, trivia, surprising details → "did_you_know"
- Boss tier list, weapon ranking, who wins → "comparison"
- Lore theory, story explained, plot twist → "theory"
- Dark history, dev secrets, hidden truth, buried lore → "dark_secrets"
- Game origin story, how it was made → "story_time"
- Easter eggs, hidden rooms, secret references → "easter_eggs"
- Top 10 games, best builds, ranking → "top_list"
"""

_sem_embedder = None


def _get_sem_embedder():
    """Lazy-load sentence-transformers model. Returns None on failure."""
    global _sem_embedder
    if _sem_embedder is not None:
        return _sem_embedder
    try:
        import logging as _logging
        _logging.getLogger("sentence_transformers").setLevel(_logging.WARNING)
        _logging.getLogger("huggingface_hub").setLevel(_logging.WARNING)
        from sentence_transformers import SentenceTransformer
        _sem_embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return _sem_embedder
    except Exception:
        return None


class TrendedTopic(BaseModel):
    title: str
    trending_reason: str
    video_angle: str
    search_queries: list[str] = Field(default_factory=list)
    skill_id: str = "did_you_know"
    hook_idea: str = ""
    urgency_score: float = 0.4
    score: float = 0.0
    sources: list[str] = Field(default_factory=list)


def _trending_cfg() -> dict:
    from src.agent_config import load_agent_settings
    cfg = dict(load_agent_settings().get("trending", {}))
    defaults = {
        "cache_ttl_secs": 86400,
        "cache_eviction_secs": 172800,
        "cache_sim_threshold": 0.88,
        "dedup_jaccard_threshold": 0.5,
        "dedup_lock_secs": 604800,
        "summarization_temperature": 0.3,
        "brainstorm_temperature": 0.6,
        "score_weights": {
            "anilist": 0.45, "jikan": 0.25, "reddit": 0.20, "popularity_norm": 0.10,
        },
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    return cfg


def fetch_trending_topics(category: str = "anime", limit: int = 10) -> list[TrendedTopic]:
    """Discover trending topics, brainstorm video angles, filter dupes.

    Returns up to `limit` TrendedTopic objects sorted by score.
    Never raises.
    """
    try:
        candidates = _get_candidates(category)
    except Exception as exc:
        logger.warning("fetch_trending_topics: candidate fetch failed: %s", exc)
        return []

    results: list[TrendedTopic] = []
    for candidate in candidates[:8]:
        try:
            topic = _brainstorm_one(candidate)
        except Exception as exc:
            logger.warning(
                "fetch_trending_topics: brainstorm failed for %r: %s",
                candidate.get("title", ""),
                exc,
            )
            topic = _fallback_topic(candidate)

        if _check_dedup(topic.title, topic.skill_id, topic.video_angle, topic.urgency_score):
            logger.info(
                "fetch_trending_topics: skipping duplicate title=%r skill_id=%r",
                topic.title,
                topic.skill_id,
            )
            continue

        try:
            _log_generated(topic.title, topic.skill_id, topic.video_angle)
        except Exception as exc:
            logger.warning("fetch_trending_topics: trend log write failed: %s", exc)

        results.append(topic)

    results.sort(key=lambda t: t.score, reverse=True)
    return results[:limit]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_candidates(category: str) -> list[dict]:
    if category == "anime":
        from src.content_sources.anime_trends import fetch_trending_anime
        return fetch_trending_anime(limit=8)
    if category == "gaming":
        from src.content_sources.gaming_trends import fetch_trending_games
        return fetch_trending_games(limit=8)
    return []


def _normalize_title(title: str) -> str:
    lowered = title.lower()
    stripped = _PUNCT_RE.sub("", lowered)
    return " ".join(stripped.split())


def _brainstorm_one(candidate: dict) -> TrendedTopic:
    from src.agent.robust_json import extract_json_dict
    title = candidate.get("title") or ""
    synopsis = candidate.get("anilist_synopsis") or candidate.get("reddit_context") or ""
    trending_reason = candidate.get("trending_reason") or ""
    reddit_context = candidate.get("reddit_context") or ""
    genres = ", ".join(candidate.get("genres") or [])
    category = candidate.get("_category", "anime")

    dense = _summarize_candidate(title, synopsis, trending_reason, reddit_context, genres)

    cached = _get_cached_brainstorm(title, dense)
    if cached:
        logger.info("brainstorm cache hit for %r", title)
        return _topic_from_cached(title, trending_reason, cached, candidate)

    brainstorm = _call_brainstorm_llm(title, dense, trending_reason, reddit_context, genres, category)
    if not _validate_brainstorm(brainstorm, title):
        logger.warning("brainstorm validation failed for %r; using fallback", title)
        brainstorm = _fallback_brainstorm(title)

    _store_cached_brainstorm(title, dense, brainstorm)
    return _topic_from_brainstorm(title, trending_reason, brainstorm, candidate)


def _summarize_candidate(title, synopsis, trending_reason, reddit_context, genres) -> str:
    """Use cheap 8b model to summarize. Returns dense 80-120 word paragraph."""
    try:
        template = _load_prompt("trend_summarize.txt")
        from src.llm_client import chat_completion
        raw = chat_completion(
            system=template.format(
                title=title, synopsis=synopsis[:1000],
                trending_reason=trending_reason, reddit_context=reddit_context[:500],
                genres=genres,
            ),
            user="Write the summary now.",
            stage="crawl",
            temperature=0.3,
            timeout=10.0,
        )
        return raw.strip()[:1500]
    except Exception as exc:
        logger.debug("summarize failed: %s", exc)
        return f"{title} - {synopsis[:300]}"


def _call_brainstorm_llm(title, dense, trending_reason, reddit_context, genres, category):
    from src.llm_client import chat_completion
    from src.agent.robust_json import extract_json_dict
    cfg = _trending_cfg()
    template = _load_prompt("trend_brainstorm.txt")
    skill_guide = _ANIME_SKILL_GUIDE if category == "anime" else _GAMING_SKILL_GUIDE
    system = template.format(
        category=category, title=title, dense_summary=dense,
        trending_reason=trending_reason, reddit_context=reddit_context[:600],
        genres=genres, skill_guide=skill_guide,
    )
    raw = chat_completion(
        system=system, user=f"Design a trending video for: {title}",
        stage="trend_brainstorm", temperature=cfg["brainstorm_temperature"],
        timeout=25.0,
    )
    return extract_json_dict(raw, required_keys=["video_angle", "search_queries", "skill_id"]) or {}


def _validate_brainstorm(parsed: dict, title: str) -> bool:
    """Reject empty / hallucinated / off-topic brainstorms."""
    if not parsed:
        return False
    angle = str(parsed.get("video_angle", "")).strip()
    queries = parsed.get("search_queries", [])
    if len(angle) < 20:
        return False
    if not isinstance(queries, list) or len(queries) < 3:
        return False
    tl = title.lower()
    combined = (angle + " " + " ".join(str(q) for q in queries)).lower()
    if tl not in combined and not any(w in combined for w in tl.split()[:2] if len(w) > 3):
        return False
    return True


def _fallback_brainstorm(title: str) -> dict:
    return {
        "video_angle": f"Top facts about {title}",
        "search_queries": [f"{title} facts", f"{title} explained", f"{title} hidden details"],
        "skill_id": "did_you_know",
        "hook_idea": f"What you didn't know about {title}.",
        "reasoning": "fallback",
    }


def _topic_from_brainstorm(title, trending_reason, parsed, candidate):
    return TrendedTopic(
        title=title,
        trending_reason=trending_reason,
        video_angle=str(parsed.get("video_angle") or title).strip(),
        search_queries=_coerce_str_list(parsed.get("search_queries")),
        skill_id=str(parsed.get("skill_id") or "did_you_know").strip() or "did_you_know",
        hook_idea=str(parsed.get("hook_idea") or "").strip(),
        urgency_score=float(candidate.get("urgency_score") or 0.4),
        score=float(candidate.get("score") or 0.0),
        sources=list(candidate.get("sources") or []),
    )


def _topic_from_cached(title, trending_reason, cached, candidate):
    return TrendedTopic(
        title=title,
        trending_reason=trending_reason,
        video_angle=str(cached.get("video_angle") or title).strip(),
        search_queries=_coerce_str_list(cached.get("search_queries")),
        skill_id=str(cached.get("skill_id") or "did_you_know").strip(),
        hook_idea=str(cached.get("hook_idea") or "").strip(),
        urgency_score=float(candidate.get("urgency_score") or 0.4),
        score=float(candidate.get("score") or 0.0),
        sources=list(candidate.get("sources") or []) + ["cache"],
    )


def _embed_text(text: str) -> list[float] | None:
    """Embed text to a 384-dim vector. Returns None on failure."""
    model = _get_sem_embedder()
    if model is None:
        return None
    try:
        vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return vec.tolist()
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two unit-normalized vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    return max(-1.0, min(1.0, dot))


def _load_sem_cache() -> dict:
    if not BRAINSTORM_CACHE.exists():
        return {"entries": []}
    try:
        data = json.loads(BRAINSTORM_CACHE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return data
    except Exception:
        pass
    return {"entries": []}


def _save_sem_cache(cache: dict) -> None:
    try:
        BRAINSTORM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        BRAINSTORM_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _get_cached_brainstorm(title: str, dense_summary: str) -> dict | None:
    """Return cached brainstorm if a semantically similar show was cached within TTL."""
    cfg = _trending_cfg()
    threshold = cfg["cache_sim_threshold"]
    ttl = cfg["cache_ttl_secs"]

    query_text = f"{title} {dense_summary[:150]}".strip()
    query_emb = _embed_text(query_text)

    cache = _load_sem_cache()
    now = time.time()

    for entry in cache.get("entries", []):
        if (now - float(entry.get("cached_at", 0))) >= ttl:
            continue
        stored_emb = entry.get("embedding")
        if query_emb and stored_emb:
            sim = _cosine(query_emb, stored_emb)
            if sim >= threshold:
                logger.info(
                    "brainstorm sem-cache hit: %r ~ %r (sim=%.3f)",
                    title, entry.get("title", "?"), sim,
                )
                return entry.get("brainstorm")
        else:
            if _normalize_title(entry.get("title", "")) == _normalize_title(title):
                return entry.get("brainstorm")
    return None


def _store_cached_brainstorm(title: str, dense_summary: str, brainstorm: dict) -> None:
    """Store brainstorm result with embedding. Evict entries older than configured threshold."""
    try:
        cfg = _trending_cfg()
        evict = cfg["cache_eviction_secs"]

        query_text = f"{title} {dense_summary[:150]}".strip()
        embedding = _embed_text(query_text)

        cache = _load_sem_cache()
        now = time.time()

        cache["entries"] = [
            e for e in cache.get("entries", [])
            if (now - float(e.get("cached_at", 0))) < evict
        ]

        cache["entries"].append({
            "title": title,
            "embedding": embedding,
            "brainstorm": brainstorm,
            "cached_at": now,
        })

        _save_sem_cache(cache)
    except Exception as exc:
        logger.warning("_store_cached_brainstorm failed: %s", exc)


def _fallback_topic(candidate: dict[str, Any]) -> TrendedTopic:
    """Build a safe default TrendedTopic when LLM call fails."""
    title = candidate.get("title") or "Unknown"
    return TrendedTopic(
        title=title,
        trending_reason=candidate.get("trending_reason") or "",
        video_angle=f"Top facts about {title}",
        search_queries=[f"{title} anime facts", f"{title} explained"],
        skill_id="did_you_know",
        hook_idea="",
        urgency_score=float(candidate.get("urgency_score") or 0.4),
        score=float(candidate.get("score") or 0.0),
        sources=list(candidate.get("sources") or []),
    )


def _parse_llm_json(raw: str) -> dict[str, Any]:
    """Extract and parse JSON from LLM response. Returns {} on failure."""
    raw = raw.strip()
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = _JSON_RE.search(raw)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

    return {}


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _load_trend_log() -> dict[str, Any]:
    """Load trend log, creating it if absent."""
    if not TREND_LOG.exists():
        return {"generated": []}
    try:
        data = json.loads(TREND_LOG.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("generated"), list):
            return data
    except Exception:
        pass
    return {"generated": []}


def _save_trend_log(log: dict[str, Any]) -> None:
    TREND_LOG.parent.mkdir(parents=True, exist_ok=True)
    TREND_LOG.write_text(
        json.dumps(log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _angle_similar(a: str, b: str, threshold: float = 0.5) -> bool:
    """Jaccard similarity on word tokens — True if angles are too close."""
    wa = set(_PUNCT_RE.sub("", a.lower()).split())
    wb = set(_PUNCT_RE.sub("", b.lower()).split())
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


def _check_dedup(title: str, skill_id: str, video_angle: str = "", urgency_score: float = 0.4) -> bool:
    """Return True if this topic should be skipped as a duplicate."""
    if urgency_score >= 1.0:
        return False

    cfg = _trending_cfg()
    threshold = cfg["dedup_jaccard_threshold"]
    lock = cfg["dedup_lock_secs"]

    log = _load_trend_log()
    norm = _normalize_title(title)
    now = time.time()
    for entry in log.get("generated") or []:
        if _normalize_title(entry.get("title") or "") != norm:
            continue
        if (now - float(entry.get("timestamp") or 0)) >= lock:
            continue
        if entry.get("skill_id") != skill_id:
            continue
        if _angle_similar(entry.get("video_angle") or "", video_angle, threshold):
            return True
    return False


def _log_generated(title: str, skill_id: str, video_angle: str) -> None:
    """Append a generated entry to the trend log."""
    log = _load_trend_log()
    log["generated"].append({
        "title": title,
        "skill_id": skill_id,
        "video_angle": video_angle,
        "timestamp": time.time(),
        "date": datetime.now(timezone.utc).isoformat(),
    })
    _save_trend_log(log)
