"""Research Agent — tier-aware, grounded fact extraction pipeline.

Pipeline:
    plan (with aliases) → tiered acquisition → chunk + enrich → hybrid recall
    → parent-doc retrieval → grounded extraction → grounding verification
    → authority-weighted dedup & scoring → quality floor → ResearchResult

Keeps existing search / wiki / crawl helpers unchanged; rewrites the
orchestration, extraction, and scoring layers.
"""

import gc
import json
import logging
import re
import time
from pathlib import Path as _Path
from typing import Any, Callable, Optional

from ..agent_config import load_agent_settings, research_settings
from .authority_registry import classify as classify_authority, tier_weight
from .contracts import (
    ChunkRecord,
    CoverageReport,
    GroundedFact,
    ResearchResult,
    SourceRecord,
)
from .entity_sanitizer import forbidden_entities, is_contaminated
from .grounding import is_grounded, topic_mentioned
from .robust_json import extract_json_dict, extract_json_list

logger = logging.getLogger(__name__)

_PROMPT_DIR = _Path(__file__).parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def _load_extraction_hint(skill_id: str) -> str:
    """Load extraction_hint from skill JSON. Falls back to empty string."""
    if not skill_id:
        return ""
    try:
        skill_path = _Path(__file__).parent.parent.parent / "skills" / f"{skill_id}.json"
        import json as _json
        data = _json.loads(skill_path.read_text(encoding="utf-8"))
        return data.get("extraction_hint", "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Module-level static data
# ---------------------------------------------------------------------------
_QUERY_TEMPLATES = [
    "{topic} hidden details",
    "{topic} dark facts",
    "{topic} things you didn't know",
    "{topic} controversy reddit",
    "{topic} why everyone is talking about",
    "{topic} fan theory memes",
    "{topic} easter eggs secrets",
    "{topic} unpopular opinions reddit",
    "{topic} the tragedy of",
    "{topic} why people hate",
]

_ENGLISH_QUERY_TEMPLATES = [
    "{eng} hidden details things you didn't know",
    "{eng} dark facts secrets",
    "{eng} fan theory controversy",
    "{eng} creator interview behind the scenes",
    "{eng} easter eggs cut content",
]

_SKIP_DOMAINS = {
    "youtube.com", "tiktok.com", "pinterest.com", "instagram.com",
    "twitter.com", "x.com", "facebook.com", "wikipedia.org",
    "amazon.com", "ebay.com", "reddit.com/gallery",
}


# ---------------------------------------------------------------------------
# Helpers (kept from original)
# ---------------------------------------------------------------------------

def _extract_english_query(topic: str) -> str:
    """Extract English/romanized proper nouns from a non-English topic string."""
    tokens = topic.split()
    stopwords = {
        "trong", "cua", "ve", "va", "nhan", "vat", "la", "mot", "nhung",
        "the", "and", "of", "in", "from", "about", "with", "for",
        "su", "that", "den", "toi", "bi", "an", "dang", "sau", "bo", "phim",
        "hay", "lam", "video",
    }
    pure_ascii = []
    for token in tokens:
        # Guard on the ORIGINAL token: if any char is non-ASCII the whole token
        # is a diacritic/CJK word and must be excluded, even after stripping.
        if not all(ord(c) < 128 for c in token):
            continue
        clean = re.sub(r"[^A-Za-z0-9:'\-]", "", token)
        if clean and len(clean) >= 2:
            if clean.lower() not in stopwords:
                pure_ascii.append(clean)
    return " ".join(pure_ascii).strip()


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting and clean HTML artifacts for TTS output."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[A-ZÀ-Ỹ][^:]{2,30}:\s*", "", text).strip()
    text = text.replace("`", "")
    text = _clean_html_text(text)
    return text.strip()


def _clean_html_text(text: str) -> str:
    """Clean HTML artifacts from search snippets and crawled text."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\x02", " ").replace("\x03", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _emit_progress(emit: Optional[Callable], phase: str, message: str):
    if emit:
        try:
            emit({"phase": phase, "message": message})
        except Exception:
            pass


def _split_paragraphs(text: str, max_chunk: int = 1500) -> list[str]:
    """Split text into paragraph-grouped chunks up to max_chunk chars."""
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p and p.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) < 80 and not current:
            current = para
            continue
        if not current:
            current = para
            continue
        if len(current) + 2 + len(para) <= max_chunk:
            current = current + "\n\n" + para
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= 80]


# ---------------------------------------------------------------------------
# RAG helper
# ---------------------------------------------------------------------------

def _try_get_rag(topic: str):
    """Return RagStore if available, else None."""
    try:
        from .rag_store import RagStore
        cfg = research_settings()
        return RagStore(topic, ttl_secs=cfg["rag_cache_ttl_secs"])
    except Exception as exc:
        logger.debug("RagStore unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Phase 1: Multi-query search (kept untouched per rules)
# ---------------------------------------------------------------------------

def _phase_search(
    topic: str,
    search_queries: list[str],
    language: str,
    emit: Optional[Callable],
) -> tuple[list[dict], list[dict]]:
    """Run multiple DDG queries, collect snippets + URLs.

    Returns (all_search_results, snippet_chunks).
    """
    from ..content_sources.duckduckgo_source import search_duckduckgo

    cfg = research_settings()
    max_ddg = cfg["max_ddg_per_query"]

    queries = list(search_queries) if search_queries else []

    if len(queries) < 5:
        for tmpl in _QUERY_TEMPLATES:
            q = tmpl.format(topic=topic)
            if q not in queries:
                queries.append(q)
            if len(queries) >= 8:
                break

    english_topic = _extract_english_query(topic)
    if english_topic and not language.startswith("en"):
        for tmpl in _ENGLISH_QUERY_TEMPLATES:
            q = tmpl.format(eng=english_topic)
            if q not in queries:
                queries.append(q)
            if len(queries) >= 12:
                break

    _emit_progress(emit, "research", f"Searching with {len(queries)} queries...")

    all_results: list[dict] = []
    snippet_chunks: list[dict] = []
    seen_urls: set[str] = set()

    for i, query in enumerate(queries):
        _emit_progress(emit, "research", f"Search [{i+1}/{len(queries)}]: {query[:50]}...")
        try:
            search_lang = language
            if query and all(ord(c) < 128 for c in query.replace(" ", "").replace("-", "").replace("'", "")):
                search_lang = "en-US"

            results = search_duckduckgo(query, language=search_lang, max_results=max_ddg)
            for r in results:
                url = r.get("href", r.get("url", ""))
                snippet = r.get("body", r.get("snippet", ""))
                title = r.get("title", "")

                if url and url not in seen_urls:
                    seen_urls.add(url)
                    r["_search_query"] = query
                    all_results.append(r)

                if snippet and len(snippet) > 50:
                    clean_snippet = _clean_html_text(snippet)
                    if len(clean_snippet) > 50:
                        snippet_chunks.append({
                            "text": clean_snippet[:500],
                            "title": title,
                            "source_url": url,
                            "url": url,
                            "source": "duckduckgo",
                        })
        except Exception as exc:
            logger.debug("DDG search failed for '%s': %s", query, exc)

    _emit_progress(emit, "research", f"Found {len(all_results)} unique URLs, {len(snippet_chunks)} snippets")
    return all_results, snippet_chunks


# ---------------------------------------------------------------------------
# Phase 2: Wikipedia (kept untouched per rules)
# ---------------------------------------------------------------------------

def _phase_wiki(
    topic: str,
    language: str,
    emit: Optional[Callable],
) -> list[dict]:
    """Fetch Wikipedia in target language + English. Returns section chunks."""
    from ..content_sources.wikipedia_source import fetch_wikipedia_draft

    wiki_chunks: list[dict] = []

    _emit_progress(emit, "research", "Fetching Wikipedia...")
    try:
        wiki = fetch_wikipedia_draft(topic, language_code=language, max_blocks=5)
        sections = wiki.get("source_draft", {}).get("sections", [])
        for sec in sections:
            text = sec.get("text", "")
            if text and len(text) > 50:
                wiki_chunks.append({
                    "text": text[:2000],
                    "title": sec.get("title", ""),
                    "source_url": sec.get("source_url", ""),
                    "url": sec.get("source_url", ""),
                    "source": "wikipedia",
                })
    except Exception as exc:
        logger.debug("Wikipedia fetch failed (%s): %s", language, exc)

    if not language.startswith("en"):
        english_topic = _extract_english_query(topic)
        if english_topic:
            _emit_progress(emit, "research", f"Fetching English Wikipedia for '{english_topic}'...")
            try:
                wiki_en = fetch_wikipedia_draft(english_topic, language_code="en-US", max_blocks=5)
                sections_en = wiki_en.get("source_draft", {}).get("sections", [])
                for sec in sections_en:
                    text = sec.get("text", "")
                    if text and len(text) > 50:
                        wiki_chunks.append({
                            "text": text[:2000],
                            "title": sec.get("title", ""),
                            "source_url": sec.get("source_url", ""),
                            "url": sec.get("source_url", ""),
                            "source": "wikipedia",
                        })
            except Exception as exc:
                logger.debug("English Wikipedia fetch failed: %s", exc)

    _emit_progress(emit, "research", f"Wikipedia: {len(wiki_chunks)} sections")
    return wiki_chunks


# ---------------------------------------------------------------------------
# Crawl soft floor (kept untouched per rules)
# ---------------------------------------------------------------------------

def _crawl_with_soft_floor(
    urls: list,
    query: str,
    base_threshold: float,
    min_pages: int = 3,
) -> list[dict]:
    """Wrap crawl_with_bm25 with a soft floor: if fewer than min_pages are
    returned, lower the threshold by 0.3 and retry (up to 2 reductions)."""
    from ..content_sources.crawl4ai_source import crawl_with_bm25

    threshold = base_threshold
    result = crawl_with_bm25(urls, query=query, bm25_threshold=threshold)
    for _ in range(2):
        if len(result) >= min_pages:
            break
        new_threshold = round(threshold - 0.3, 1)
        logger.warning(
            "BM25 threshold %.1f returned %d pages — lowering to %.1f",
            threshold,
            len(result),
            new_threshold,
        )
        threshold = new_threshold
        result = crawl_with_bm25(urls, query=query, bm25_threshold=threshold)
    return result


# ---------------------------------------------------------------------------
# Evaluate + Reflect (gap-filling)
# ---------------------------------------------------------------------------

def _phase_evaluate(
    topic: str,
    fact_count: int,
    target: int,
    user_prompt: str = "",
    must_cover: list[str] | None = None,
    covered_angles: str = "",
) -> list[str]:
    """Ask LLM which key angles are missing. Returns list of missing angle strings."""
    try:
        from ..llm_client import chat_completion
        response = chat_completion(
            system="You are a research gap detector. Return ONLY valid JSON.",
            user=_load_prompt("research_evaluate.txt").format(
                topic=topic,
                fact_count=fact_count,
                target=target,
                user_prompt=user_prompt or topic,
                must_cover=", ".join(must_cover or []) or "(not specified)",
                covered_angles=covered_angles or "(none yet)",
            ),
            stage="research_eval",
            temperature=0.1,
            timeout=15.0,
        )
        data = extract_json_dict(response) or {}
        if data.get("sufficient", True):
            return []
        return [str(a).strip() for a in data.get("missing_angles", []) if str(a).strip()]
    except Exception as exc:
        logger.debug("Evaluate phase failed: %s", exc)
        return []


def _phase_reflect_crawl(
    topic: str,
    missing_angles: list[str],
    language: str,
    existing_urls: set[str],
    emit: Optional[Callable],
    extraction_hint: str = "",
) -> list[dict]:
    """Run one extra DDG+crawl pass for the missing angles. Returns raw pages."""
    if not missing_angles:
        return []

    cfg = research_settings()
    extra_queries = [f"{topic} {angle}" for angle in missing_angles[:3]]
    _emit_progress(emit, "research", f"Reflect: searching {len(extra_queries)} gap queries...")

    search_results, _ = _phase_search(topic, extra_queries, language, emit)

    new_urls = []
    for r in search_results:
        url = r.get("href", r.get("url", ""))
        if not url or url in existing_urls:
            continue
        if any(d in url.lower() for d in _SKIP_DOMAINS):
            continue
        new_urls.append(url)
        existing_urls.add(url)

    if not new_urls:
        _emit_progress(emit, "research", "Reflect: no new URLs found")
        return []

    _emit_progress(emit, "research", f"Reflect: crawling {min(len(new_urls), cfg['max_crawl_pages'])} new pages...")
    try:
        english_topic = _extract_english_query(topic) or topic
        # Use the first uncovered angle as the BM25 query so the crawl targets
        # the specific gap, not just the topic name.
        crawl_query = f"{english_topic} {missing_angles[0]}" if missing_angles else english_topic
        new_pages = _crawl_with_soft_floor(
            new_urls[:cfg["max_crawl_pages"]],
            query=crawl_query,
            base_threshold=cfg["bm25_threshold"],
        )
    except Exception as exc:
        logger.debug("Reflect crawl failed: %s", exc)
        return []

    return new_pages or []


# ---------------------------------------------------------------------------
# Last-resort LLM fallback (legacy — unused by new grounded path)
# ---------------------------------------------------------------------------

def _last_resort_llm(topic: str, language: str) -> list[dict]:
    """When all search fails, ask LLM directly. Last resort."""
    try:
        from ..llm_client import chat_completion

        _fallback_tmpl = _load_prompt("research_fallback.txt")
        _vi_part, _, _en_part = _fallback_tmpl.partition("=== en ===")
        _vi_part = _vi_part.replace("=== vi ===", "").strip()
        _en_part = _en_part.strip()
        if language.startswith("vi"):
            prompt = _vi_part.format(topic=topic)
        else:
            prompt = _en_part.format(topic=topic)

        answer = chat_completion(
            system="You are a research assistant. Give specific, surprising facts only.",
            user=prompt,
            stage="research",
            temperature=0.4,
            timeout=20.0,
        )

        if not answer or len(answer) < 50:
            return []

        facts = []
        lines = answer.strip().split("\n")
        for line in lines:
            text = line.strip()
            text = re.sub(r"^\d+[\.\)]\s*", "", text).strip()
            text = re.sub(r"^[-•]\s*", "", text).strip()
            text = _strip_markdown(text)
            if len(text) > 20:
                facts.append({
                    "fact_text": text,
                    "source": "llm",
                    "score": 0.4,
                })

        if not facts and len(answer.strip()) > 50:
            facts.append({
                "fact_text": _strip_markdown(answer.strip()[:1000]),
                "source": "llm",
                "score": 0.3,
            })

        return facts
    except Exception as exc:
        logger.warning("Last-resort LLM failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Query expansion (new — uses research_expand_queries.txt)
# ---------------------------------------------------------------------------

def _expand_queries_llm(
    topic: str,
    base_queries: list[str],
    n_desired: int = 4,
    topic_category: str = "",
    topic_aliases: list[str] | None = None,
    missing_angles: list[str] | None = None,
) -> list[str]:
    """Add up to `n_desired` LLM-generated queries to `base_queries`."""
    existing = [q for q in (base_queries or []) if q]
    if len(existing) >= 6:
        return existing
    try:
        template = _load_prompt("research_expand_queries.txt")
    except Exception:
        return existing
    try:
        prompt = template.format(
            topic=topic,
            topic_category=topic_category or "",
            topic_aliases=", ".join(topic_aliases or [topic]),
            existing_queries=", ".join(existing) or "(none)",
            n_queries=n_desired,
            missing_angles=", ".join(missing_angles or []) or "(none)",
        )
    except Exception as exc:
        logger.debug("expand_queries template format failed: %s", exc)
        return existing
    try:
        from ..llm_client import chat_completion
        raw = chat_completion(
            system=prompt,
            user=f"Generate queries for: {topic}",
            stage="research_eval",
            temperature=0.3,
            timeout=15.0,
        )
    except Exception:
        return existing
    new_list = extract_json_list(raw) or []
    more = [str(q).strip() for q in new_list if isinstance(q, str) and q.strip()]
    combined = existing + [q for q in more if q not in existing]
    return combined[:10]


# ---------------------------------------------------------------------------
# Page → Tiered SourceRecord
# ---------------------------------------------------------------------------

def _domain_of(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _classify_pages_by_tier(pages: list[dict], topic_aliases: list[str]) -> list[SourceRecord]:
    """Convert legacy page dicts to SourceRecord, attaching authority tier."""
    out: list[SourceRecord] = []
    cfg = research_settings()
    max_chars = cfg["page_text_max_chars"]
    for p in pages:
        url = p.get("url") or p.get("source_url") or ""
        if not url:
            continue
        tier = classify_authority(url, topic_aliases=topic_aliases)
        domain = _domain_of(url)
        out.append(SourceRecord(
            url=url,
            domain=domain,
            authority_tier=tier,
            title=p.get("title") or "",
            fetched_at=time.time(),
            raw_text=(p.get("text") or "")[:max_chars],
            chunks=[],
        ))
    return out


# ---------------------------------------------------------------------------
# Parent-aware chunking
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,3}\s+.+|[A-Z][A-Z0-9 ]{4,80})$", re.MULTILINE)


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _split_paragraph_at_sentences(para: str, chunk_size: int, chunk_min: int, overlap: int) -> list[str]:
    """Split an oversized paragraph into sentence-boundary chunks."""
    sentences = _SENTENCE_END_RE.split(para)
    result: list[str] = []
    buf = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(buf) + len(sent) + 1 <= chunk_size:
            buf = f"{buf} {sent}".strip() if buf else sent
        else:
            if len(buf) >= chunk_min:
                result.append(buf)
            tail = buf[-overlap:] if overlap and buf else ""
            buf = f"{tail} {sent}".strip() if tail else sent
    if len(buf) >= chunk_min:
        result.append(buf)
    return result or [para]


def _chunks_with_enrichment(source: SourceRecord, chunk_size: int,
                            chunk_min: int, overlap: int) -> list[ChunkRecord]:
    """Split a source's raw_text into enriched chunks.

    Each chunk carries preceding_context (last ~120 chars of previous chunk),
    following_context (first ~120 chars of next chunk), and section_heading
    (most recent heading-like line above the chunk).

    Long paragraphs are split at sentence boundaries (semantic chunking) rather
    than arbitrary character offsets, keeping related sentences together.
    """
    text = (source.raw_text or "").strip()
    if not text:
        return []

    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[tuple[str, str]] = []
    current_heading = ""
    buf = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if _HEADING_RE.match(para) and len(para) <= 100:
            current_heading = para.lstrip("# ").strip()
            continue
        # Oversized paragraph: split at sentence boundaries first
        if len(para) > chunk_size:
            if len(buf) >= chunk_min:
                chunks.append((current_heading, buf))
                buf = ""
            for sub in _split_paragraph_at_sentences(para, chunk_size, chunk_min, overlap):
                chunks.append((current_heading, sub))
            continue
        if len(buf) + len(para) + 2 <= chunk_size:
            buf = f"{buf}\n\n{para}".strip() if buf else para
        else:
            if len(buf) >= chunk_min:
                chunks.append((current_heading, buf))
            tail = buf[-overlap:] if overlap and buf else ""
            buf = f"{tail}\n\n{para}".strip() if tail else para
    if len(buf) >= chunk_min:
        chunks.append((current_heading, buf))

    out: list[ChunkRecord] = []
    for i, (heading, body) in enumerate(chunks):
        prev = chunks[i - 1][1] if i > 0 else ""
        nxt = chunks[i + 1][1] if i + 1 < len(chunks) else ""
        out.append(ChunkRecord(
            chunk_id=f"tmp:{i}",
            text=body,
            section_heading=heading,
            preceding_context=prev[-120:] if prev else "",
            following_context=nxt[:120] if nxt else "",
            source_url=source.url,
            page_title=source.title,
            authority_tier=source.authority_tier,
        ))
    source.chunks = out
    return out


# ---------------------------------------------------------------------------
# Parent-window aware extraction
# ---------------------------------------------------------------------------

def _extract_from_chunk_context(
    chunk: dict,
    parent_window_text: str,
    topic: str,
    topic_aliases: list[str],
    extraction_hint: str,
    tier: int,
    video_goal: str = "",
    must_cover: str = "",
) -> list[dict]:
    """Ask LLM to extract facts from the parent window. LLM must provide
    verbatim_evidence for each fact."""
    try:
        prompt_template = _load_prompt("research_extract.txt")
    except Exception as exc:
        logger.error("research_extract.txt missing or broken: %s", exc)
        return []

    try:
        system_prompt = prompt_template.format(
            topic=topic,
            topic_aliases=", ".join(topic_aliases),
            source_url=chunk.get("source_url", ""),
            page_title=chunk.get("page_title", ""),
            authority_tier=tier,
            extraction_hint=extraction_hint or "Extract specific, non-obvious, surprising facts.",
            video_goal=video_goal or topic,
            must_cover=must_cover or "(not specified)",
        )
    except Exception as exc:
        logger.debug("extract prompt format failed: %s", exc)
        return []

    user_prompt = (
        f"SECTION HEADING: {chunk.get('section_heading', '')}\n\n"
        f"PARENT WINDOW:\n{parent_window_text.strip()}\n\n"
        f"Return JSON only."
    )
    try:
        from ..llm_client import extraction_completion
        raw = extraction_completion(
            system=system_prompt,
            user=user_prompt,
            temperature=0.1,
        )
    except Exception as exc:
        logger.debug("extraction LLM failure: %s", exc)
        return []

    parsed = extract_json_dict(raw, required_keys=["is_relevant", "facts"])
    if not parsed:
        logger.debug("extraction: JSON parse failed for %s", chunk.get("source_url", ""))
        return []
    if not parsed.get("is_relevant"):
        return []
    out = []
    for raw_fact in parsed.get("facts", []):
        if not isinstance(raw_fact, dict):
            continue
        claim = str(raw_fact.get("claim", "")).strip()
        evidence = str(raw_fact.get("verbatim_evidence", "")).strip()
        try:
            conf = float(raw_fact.get("confidence", 0.0) or 0.0)
        except (ValueError, TypeError):
            conf = 0.0
        if len(claim) < 20 or len(evidence) < 30 or conf < 0.4:
            continue
        out.append({
            "claim": claim,
            "verbatim_evidence": evidence,
            "confidence": conf,
            "source_url": chunk.get("source_url", ""),
            "source_domain": _domain_of(chunk.get("source_url", "")),
            "authority_tier": tier,
            "parent_text": parent_window_text,
            "angle_served": str(raw_fact.get("angle_served", "") or "").strip(),
        })
    return out


# ---------------------------------------------------------------------------
# Grounding + scoring
# ---------------------------------------------------------------------------

def _verify_grounding(
    raw_facts: list[dict],
    topic_aliases: list[str],
    fuzzy_threshold: float,
) -> tuple[list[GroundedFact], dict]:
    """Run each candidate fact through grounding + topic-match checks.
    Returns (kept_facts, stats)."""
    stats = {"failed_grounding": 0, "failed_topic": 0}
    kept: list[GroundedFact] = []
    seq = 0
    for rf in raw_facts:
        claim = rf["claim"]
        evidence = rf["verbatim_evidence"]
        parent = rf.get("parent_text", "") or evidence
        tier = rf["authority_tier"]
        grounded = is_grounded(evidence, parent, fuzzy_threshold=fuzzy_threshold, tier=tier)
        if not grounded:
            stats["failed_grounding"] += 1
            continue
        topic_ok = topic_mentioned(claim, topic_aliases)
        if not topic_ok and tier > 1:
            # tier-1 sources are trusted to be on-topic even without alias mention
            stats["failed_topic"] += 1
            continue
        seq += 1
        final_score = rf["confidence"] * tier_weight(tier)
        kept.append(GroundedFact(
            fact_id=f"F{seq:03d}",
            claim=claim,
            verbatim_evidence=evidence,
            source_url=rf["source_url"],
            source_domain=rf["source_domain"],
            authority_tier=tier,
            extraction_confidence=rf["confidence"],
            final_score=final_score,
            grounded=True,
            topic_match=topic_ok,
            reason_tags=["grounded", f"tier{tier}"],
            angle_served=str(rf.get("angle_served", "") or "").strip(),
        ))
    return kept, stats


def _score_and_rank_facts(facts: list[GroundedFact], max_final: int) -> list[GroundedFact]:
    """Sort by final_score descending, Jaccard-dedup, truncate."""
    sorted_facts = sorted(facts, key=lambda f: -f.final_score)
    kept: list[GroundedFact] = []
    for f in sorted_facts:
        dup = False
        for k in kept:
            if _jaccard(k.claim, f.claim) >= 0.5:
                dup = True
                break
        if not dup:
            kept.append(f)
        if len(kept) >= max_final:
            break
    for i, f in enumerate(kept, start=1):
        f.fact_id = f"F{i:03d}"
    return kept


_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "of", "in", "on", "at", "to", "and", "or",
    "with", "for", "from", "this", "that", "as", "was", "were", "by",
})


def _jaccard(a: str, b: str) -> float:
    wa = {w for w in re.findall(r"[\w']+", a.lower()) if w not in _STOPWORDS}
    wb = {w for w in re.findall(r"[\w']+", b.lower()) if w not in _STOPWORDS}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ---------------------------------------------------------------------------
# Quality floor
# ---------------------------------------------------------------------------

class InsufficientResearchError(Exception):
    """Raised when the research pipeline cannot produce the minimum
    number of grounded, authority-vetted facts."""


def _angle_has_coverage(angle: str, facts: list[GroundedFact]) -> bool:
    """Return True iff at least one fact shares a ≥4-char token with the angle text."""
    angle_tokens = set(re.findall(r"[a-z]{4,}", angle.lower()))
    if not angle_tokens:
        return True  # can't check — don't block
    for f in facts:
        combined = (f.claim + " " + f.verbatim_evidence).lower()
        if angle_tokens & set(re.findall(r"[a-z]{4,}", combined)):
            return True
    return False


def _check_quality_floor(
    facts: list[GroundedFact],
    cfg: dict,
    warnings: list[str],
    must_cover_list: list[str] | None = None,
) -> tuple[bool, bool]:
    """Returns (min_fact_met, min_tier_diversity_met)."""
    min_grounded = cfg["min_grounded_facts"]
    min_diversity = cfg["min_tier_diversity"]
    require_tier = cfg["require_min_tier"]

    eligible = [f for f in facts if f.grounded and f.authority_tier <= require_tier]
    fact_ok = len(eligible) >= min_grounded
    domain_set = {f.source_domain for f in eligible}
    diversity_ok = len(domain_set) >= min_diversity
    if not fact_ok:
        warnings.append(
            f"Quality floor: only {len(eligible)} grounded facts from tier<={require_tier} "
            f"sources (minimum {min_grounded})."
        )
    if not diversity_ok:
        warnings.append(
            f"Quality floor: only {len(domain_set)} distinct domains at tier<={require_tier} "
            f"(minimum {min_diversity})."
        )
    # Angle-coverage check: every must_cover angle needs ≥1 supporting fact
    for angle in (must_cover_list or []):
        if not _angle_has_coverage(angle, eligible):
            warnings.append(f"Quality floor: no facts covering angle '{angle[:60]}'.")
            fact_ok = False
    return fact_ok, diversity_ok


# ---------------------------------------------------------------------------
# Legacy low-quality URL filter
# ---------------------------------------------------------------------------

_LOW_QUALITY_DOMAINS = {
    ".store", "merchandise", "merch", "shop.", "buy.", "amazon.",
    "ebay.", "etsy.", "redbubble.", "teepublic.", "aliexpress.",
}


def _is_low_quality_url(url: str) -> bool:
    url_lower = url.lower()
    return any(marker in url_lower for marker in _LOW_QUALITY_DOMAINS)


def _prepare_crawl_urls(search_results: list[dict], skip_domains: set[str]) -> list[str]:
    urls = []
    seen = set()
    for r in search_results:
        url = r.get("url") or r.get("href") or ""
        if not url or url in seen:
            continue
        dom = _domain_of(url)
        if any(sd in dom for sd in skip_domains):
            continue
        if _is_low_quality_url(url):
            continue
        seen.add(url)
        urls.append(url)
    return urls


# ---------------------------------------------------------------------------
# Retrieved-chunk extraction (concurrent, parent-window-aware)
# ---------------------------------------------------------------------------

def _build_batch_enriched(top_chunks: list[dict], rag, cfg: dict) -> list[dict]:
    """Enrich each retrieved chunk with its parent-window text + metadata.

    Parent windows are stitched from neighbouring chunks via
    ``RagStore.get_parent_window`` when the chunk came from a real RAG
    collection; inline fallback chunks just use their own text.
    """
    enriched: list[dict] = []
    for i, chunk in enumerate(top_chunks):
        window_text = chunk.get("text") or ""
        try:
            if rag and "chunk_id" in chunk and not str(chunk["chunk_id"]).startswith("inline:"):
                window = rag.get_parent_window(chunk["chunk_id"], window=cfg["parent_window_chunks"])
                if window:
                    window_text = "\n\n".join(w["text"] for w in window)
        except Exception as exc:
            logger.debug("parent_window failed for chunk %d: %s", i, exc)
        enriched.append({
            "index": i,
            "source_url": chunk.get("source_url", ""),
            "source_domain": _domain_of(chunk.get("source_url", "")),
            "page_title": chunk.get("page_title", ""),
            "section_heading": chunk.get("section_heading", ""),
            "authority_tier": int(chunk.get("authority_tier", 4) or 4),
            "window_text": window_text,
        })
    return enriched


def _extract_batch_one_call(
    enriched: list[dict],
    topic: str,
    aliases: list[str],
    extraction_hint: str,
    video_goal: str = "",
    must_cover: str = "",
) -> list[dict]:
    """Send ALL enriched chunks to the LLM in a single extraction call.

    Now uses the 'Investigator' prompt to synthesize unique facts across 
    all provided chunks.
    """
    if not enriched:
        return []
    try:
        prompt_template = _load_prompt("research_extract.txt")
    except Exception as exc:
        logger.error("research_extract.txt missing or broken: %s", exc)
        return []

    blocks: list[str] = []
    for e in enriched:
        blocks.append(
            f"=== CHUNK {e['index']} (source={e['source_url']}) ===\n"
            f"{e['window_text'].strip()}"
        )
    
    passages_text = "\n\n".join(blocks)

    try:
        system_prompt = prompt_template.format(
            topic=topic,
            video_goal=video_goal or topic,
            must_cover=must_cover or "(not specified)",
            extraction_hint=extraction_hint or "Find shocking lore and community secrets.",
            passages=passages_text
        )
    except Exception as exc:
        logger.debug("extract prompt format failed: %s", exc)
        return []

    try:
        from ..llm_client import chat_completion
        raw = chat_completion(
            system=system_prompt,
            user="Execute investigation and return JSON.",
            stage="research_extract",
            temperature=0.1,
        )
    except Exception as exc:
        logger.debug("batch extraction LLM failure: %s", exc)
        return []

    parsed = extract_json_dict(raw, required_keys=["facts"])
    if not parsed:
        return []

    idx_map = {e["index"]: e for e in enriched}
    out: list[dict] = []
    for raw_fact in parsed.get("facts", []):
        if not isinstance(raw_fact, dict):
            continue
        try:
            ci = int(raw_fact.get("chunk_index", -1))
        except (ValueError, TypeError):
            continue
        if ci not in idx_map:
            continue
        
        src = idx_map[ci]
        claim = str(raw_fact.get("claim", "")).strip()
        evidence = str(raw_fact.get("verbatim_evidence", "")).strip()
        
        if len(claim) < 20 or len(evidence) < 30:
            continue

        out.append({
            "claim": claim,
            "verbatim_evidence": evidence,
            "confidence": float(raw_fact.get("confidence", 0.7)),
            "source_url": src["source_url"],
            "source_domain": _domain_of(src["source_url"]),
            "authority_tier": src["authority_tier"],
            "parent_text": src["window_text"],
            "angle_served": str(raw_fact.get("angle_served", "") or "general").strip(),
        })
    return out


def _extract_from_retrieved(
    top_chunks: list[dict],
    rag,
    topic: str,
    aliases: list[str],
    extraction_hint: str,
    cfg: dict,
    coverage: CoverageReport,
    warnings: list[str],
    video_goal: str = "",
    must_cover: str = "",
) -> list[dict]:
    """Run extraction over retrieved chunks.

    Strategy: batch ALL retrieved chunks into a single LLM call when the
    effective extraction model has a large context window (gemma-4-31b-it
    with 256K is our current default). This collapses N extraction calls
    into 1, bypassing per-minute rate limits on free tiers.

    When `extract_concurrency > 1` and the batch call fails or returns
    zero facts, fall back to the per-chunk path for robustness.
    """
    if not top_chunks:
        return []
    try:
        _load_prompt("research_extract.txt")
    except Exception as exc:
        warnings.append(f"research_extract.txt missing: {exc}")
        return []

    enriched = _build_batch_enriched(top_chunks, rag, cfg)

    # Pass 1: angle-specific extraction (uses skill's extraction_hint)
    out = _extract_batch_one_call(enriched, topic, aliases, extraction_hint,
                                  video_goal=video_goal, must_cover=must_cover)

    # Pass 2: broad extraction — drops the narrow skill hint and asks for
    # ANY specific, non-obvious fact about the topic. Caller will still run
    # grounding + topic-match + authority weighting, so quality isn't at risk.
    min_floor = int(cfg.get("min_grounded_facts", 6))
    if len(out) < min_floor:
        broad_hint = (
            "Extract any specific, non-obvious, or surprising fact about this topic — "
            "production trivia, character backstory, behind-the-scenes decisions, "
            "creator history, sales/awards milestones, or unusual details. Skip generic "
            "plot summary but accept concrete factual claims."
        )
        extra = _extract_batch_one_call(enriched, topic, aliases, broad_hint,
                                        video_goal=video_goal, must_cover=must_cover)
        # Merge, de-dup by claim first 60 chars
        seen = {f["claim"][:60].lower() for f in out}
        for f in extra:
            key = f["claim"][:60].lower()
            if key not in seen:
                out.append(f)
                seen.add(key)

    if out:
        return out

    # Fallback: per-chunk extraction if the batch call returned nothing
    # (e.g. JSON parse error, model timeout). Concurrency stays at 1 to
    # keep us under free-tier RPM caps.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    concurrency = max(1, int(cfg.get("extract_concurrency", 1)))

    def _do_one(e):
        try:
            return _extract_from_chunk_context(
                chunk={
                    "text": e["window_text"],
                    "source_url": e["source_url"],
                    "page_title": e["page_title"],
                    "authority_tier": e["authority_tier"],
                    "section_heading": e["section_heading"],
                },
                parent_window_text=e["window_text"],
                topic=topic,
                topic_aliases=aliases,
                extraction_hint=extraction_hint,
                tier=e["authority_tier"],
                video_goal=video_goal,
                must_cover=must_cover,
            )
        except Exception as exc:
            logger.debug("fallback extract one failed: %s", exc)
            return []

    out = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for fut in as_completed([ex.submit(_do_one, e) for e in enriched]):
            out.extend(fut.result())
    return out


# ---------------------------------------------------------------------------
# Legacy result builder (returns the dict shape other modules expect)
# ---------------------------------------------------------------------------

def _build_legacy(data: dict) -> dict:
    """Build the legacy dict shape the rest of the pipeline expects."""
    facts: list[GroundedFact] = data["facts"]
    sources = data.get("sources") or []
    source_manifest = sources if isinstance(sources, list) and sources and isinstance(sources[0], SourceRecord) else []
    result = ResearchResult(
        facts=facts,
        source_manifest=source_manifest,
        coverage=data["coverage"],
        warnings=data["warnings"],
        elapsed_sec=data["elapsed"],
        cache_hit=data["cache_hit"],
    )
    legacy = result.to_legacy_dict()
    legacy["facts"] = [f.to_legacy_dict() for f in facts]
    return legacy


# Module-level handle for tests / debugging
_last_result: ResearchResult | None = None


# ---------------------------------------------------------------------------
# Persistence to content bank (legacy helper — unchanged)
# ---------------------------------------------------------------------------

def _store_to_bank(topic: str, facts: list[dict], language: str = "en-US") -> None:
    """Silently persist research facts into the content bank. Never raises."""
    try:
        import hashlib
        from ..content_bank.store import ContentBankStore
        from ..content_bank.models import FactCard

        store = ContentBankStore()
        topic_id = store.compute_topic_id(topic_query=topic, language=language)

        cards: list[FactCard] = []
        for fact in facts:
            fact_text = str(fact.get("fact_text") or "").strip()
            if not fact_text:
                continue
            if int(fact.get("authority_tier") or 4) > 2:
                continue
            raw = f"{topic_id}:{fact_text[:120]}"
            fact_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
            cards.append(FactCard(
                fact_id=fact_id,
                topic_id=topic_id,
                topic_label=topic,
                fact_text=fact_text,
                hook_text=str(fact.get("hook_text") or ""),
                source_url=str(fact.get("source_url") or ""),
                language=language,
            ))

        if cards:
            created, updated = store.upsert_facts(cards)
            logger.debug(
                "content_bank: stored %d facts for '%s' (%d new, %d updated)",
                len(cards), topic, created, updated,
            )
    except Exception as exc:
        logger.debug("content_bank store skipped: %s", exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_research(
    topic: str,
    search_queries: list[str] | None = None,
    language: str = "en-US",
    skill_id: str = "",
    emit: Optional[Callable[[dict], None]] = None,
    topic_aliases: list[str] | None = None,
    user_prompt: str = "",
    must_cover: list[str] | None = None,
    topic_category: str = "",
) -> dict[str, Any]:
    """Grounded, tier-aware research.

    Returns the legacy dict shape for backward compatibility:
      {"facts": [...], "sources_used": [...], "warnings": [...], "coverage": str}

    The internal ResearchResult is available via `_last_result` for tests.
    """
    cfg = research_settings(topic_category=topic_category)
    start = time.monotonic()
    warnings: list[str] = []
    aliases = list({a for a in (topic_aliases or []) + [topic] if a})
    extraction_hint = _load_extraction_hint(skill_id)
    must_cover_list = list(must_cover or [])
    must_cover_str = ", ".join(must_cover_list) if must_cover_list else ""
    video_goal = user_prompt or topic
    if emit is None:
        emit = lambda _e: None

    coverage = CoverageReport()

    # Sanitizer: pre-compute forbidden entities for later contamination checks
    forbidden = forbidden_entities(topic, topic_aliases=aliases)

    # --- Stage 0: RAG cache check ------------------------------------
    rag = _try_get_rag(topic)
    if rag and rag.is_cached():
        try:
            emit({"status": "rag_cache_hit", "message": f"RAG cache hit for {topic!r}"})
        except Exception:
            pass
        retrieve_queries = (list(search_queries or []) + [topic])[:5]
        cached_chunks = rag.retrieve(
            retrieve_queries,
            top_k=cfg["retrieval_top_k"],
            min_tier=cfg["require_min_tier"],
            rrf_k=cfg["rag_cache_rrf_k"],
            tier_weights={
                1: cfg["tier_1_weight"],
                2: cfg["tier_2_weight"],
                3: cfg["tier_3_weight"],
                4: cfg["tier_4_weight"],
            },
        )
        cached_facts = _extract_from_retrieved(
            cached_chunks, rag, topic, aliases, extraction_hint, cfg, coverage, warnings,
            video_goal=video_goal, must_cover=must_cover_str,
        )
        if forbidden:
            cached_facts = [rf for rf in cached_facts if not is_contaminated(rf["claim"], forbidden)[0]]
        kept, gstats = _verify_grounding(cached_facts, aliases, cfg["grounding_fuzzy_threshold"])
        coverage.facts_grounded = len(kept)
        coverage.facts_failed_grounding = gstats["failed_grounding"]
        coverage.facts_failed_topic_match = gstats["failed_topic"]
        final = _score_and_rank_facts(kept, cfg["max_final_facts"])
        _check_quality_floor(final, cfg, warnings)
        return _build_legacy({
            "facts": final,
            "coverage": coverage,
            "warnings": warnings,
            "elapsed": time.monotonic() - start,
            "cache_hit": True,
            "sources": [],
        })

    # --- Stage 1: Query expansion -------------------------------------
    expanded_queries = _expand_queries_llm(
        topic,
        list(search_queries or []),
        n_desired=cfg["n_query_expansion"],
        missing_angles=[],
    )
    try:
        emit({"status": "expanded_queries", "message": f"{len(expanded_queries)} queries"})
    except Exception:
        pass

    # --- Stage 2: Search + wiki + crawl -------------------------------
    search_results, snippet_chunks = _phase_search(topic, expanded_queries, language, emit)
    wiki_chunks = _phase_wiki(topic, language, emit)
    crawl_urls = _prepare_crawl_urls(search_results, _SKIP_DOMAINS)
    try:
        english_topic = _extract_english_query(topic) or topic
        if must_cover_list:
            # Per-angle BM25 crawl: score the URL pool separately per angle so
            # pages relevant to a specific angle aren't dropped by a topic-only query.
            angle_pages: list[dict] = []
            seen_crawl_urls: set[str] = set()
            per_angle_cap = max(2, cfg["max_crawl_pages"] // len(must_cover_list))
            url_pool = crawl_urls[:cfg["max_crawl_pages"]]
            for angle in must_cover_list:
                angle_query = f"{english_topic} {angle}"
                try:
                    pages = _crawl_with_soft_floor(url_pool, query=angle_query, base_threshold=cfg["bm25_threshold"])
                    added = 0
                    for p in pages:
                        url = p.get("url") or p.get("source_url") or ""
                        if url and url not in seen_crawl_urls:
                            seen_crawl_urls.add(url)
                            angle_pages.append(p)
                            added += 1
                            if added >= per_angle_cap:
                                break
                except Exception as exc:
                    warnings.append(f"Angle crawl failed for '{angle[:40]}': {exc}")
            crawled_pages = angle_pages
        else:
            crawled_pages = _crawl_with_soft_floor(
                crawl_urls[:cfg["max_crawl_pages"]],
                query=english_topic,
                base_threshold=cfg["bm25_threshold"],
            )
    except Exception as exc:
        warnings.append(f"Crawl failed: {exc}")
        crawled_pages = []

    all_pages = list(wiki_chunks) + list(crawled_pages) + list(snippet_chunks)
    try:
        emit({"status": "acquired", "message": f"{len(all_pages)} pages collected"})
    except Exception:
        pass

    # --- Stage 3: Tier-tag sources & enrich chunks --------------------
    sources = _classify_pages_by_tier(all_pages, aliases)
    for s in sources:
        _chunks_with_enrichment(
            s,
            chunk_size=cfg["chunk_size_chars"],
            chunk_min=cfg["chunk_min_chars"],
            overlap=cfg["chunk_overlap_chars"],
        )
    coverage.total_urls_fetched = len(sources)
    for s in sources:
        if s.authority_tier == 1:
            coverage.tier_1_count += 1
        elif s.authority_tier == 2:
            coverage.tier_2_count += 1
        elif s.authority_tier == 3:
            coverage.tier_3_count += 1
        else:
            coverage.tier_4_count += 1

    # --- Stage 4: Index chunks to RAG ---------------------------------
    chunk_dicts = []
    for s in sources:
        for j, ch in enumerate(s.chunks):
            chunk_dicts.append({
                "text": ch.text,
                "source_url": s.url,
                "page_title": s.title,
                "authority_tier": s.authority_tier,
                "section_heading": ch.section_heading,
                "preceding_context": ch.preceding_context,
                "following_context": ch.following_context,
                "chunk_idx": j,
            })
    coverage.chunks_indexed = len(chunk_dicts)
    if rag and chunk_dicts:
        rag.add_chunks(chunk_dicts)

    # --- Stage 5: Hybrid retrieve -------------------------------------
    retrieve_queries = expanded_queries[:4] + [topic]
    if rag:
        top_chunks = rag.retrieve(
            retrieve_queries,
            top_k=cfg["retrieval_top_k"],
            min_tier=cfg["require_min_tier"],
            rrf_k=cfg["rag_cache_rrf_k"],
            tier_weights={
                1: cfg["tier_1_weight"],
                2: cfg["tier_2_weight"],
                3: cfg["tier_3_weight"],
                4: cfg["tier_4_weight"],
            },
        )
    else:
        top_chunks = []

    # If RAG retrieval returned nothing, fall back to tier-sorted in-memory chunks
    if not top_chunks:
        sorted_sources = sorted(sources, key=lambda s: s.authority_tier)
        for s in sorted_sources:
            for j, ch in enumerate(s.chunks):
                top_chunks.append({
                    "chunk_id": f"inline:{s.url}:{j}",
                    "text": ch.text,
                    "source_url": s.url,
                    "page_title": s.title,
                    "authority_tier": s.authority_tier,
                    "section_heading": ch.section_heading,
                    "preceding_context": ch.preceding_context,
                    "following_context": ch.following_context,
                    "chunk_idx": j,
                })
                if len(top_chunks) >= cfg["retrieval_top_k"]:
                    break
            if len(top_chunks) >= cfg["retrieval_top_k"]:
                break
    coverage.chunks_retrieved = len(top_chunks)

    # --- Stage 6: Extract facts with parent window --------------------
    raw_facts = _extract_from_retrieved(
        top_chunks, rag, topic, aliases, extraction_hint, cfg, coverage, warnings,
        video_goal=video_goal, must_cover=must_cover_str,
    )
    coverage.facts_extracted = len(raw_facts)

    # --- Stage 6b: Foreign-franchise contamination filter -------------
    if forbidden:
        before = len(raw_facts)
        raw_facts = [rf for rf in raw_facts if not is_contaminated(rf["claim"], forbidden)[0]]
        dropped = before - len(raw_facts)
        if dropped:
            logger.info("contamination filter dropped %d raw facts", dropped)

    # --- Stage 7: Ground + score --------------------------------------
    grounded, gstats = _verify_grounding(raw_facts, aliases, cfg["grounding_fuzzy_threshold"])
    coverage.facts_grounded = len(grounded)
    coverage.facts_failed_grounding = gstats["failed_grounding"]
    coverage.facts_failed_topic_match = gstats["failed_topic"]
    final = _score_and_rank_facts(grounded, cfg["max_final_facts"])

    # --- Stage 8: Quality floor + gap-fill loop -----------------------
    fact_ok, diversity_ok = _check_quality_floor(final, cfg, warnings, must_cover_list=must_cover_list)
    coverage.min_fact_threshold_met = fact_ok
    coverage.min_tier_diversity_met = diversity_ok
    iteration = 1
    while (not fact_ok) and iteration < cfg["max_research_iterations"]:
        try:
            emit({"status": "reflecting", "message": f"Quality floor unmet; iteration {iteration+1}"})
        except Exception:
            pass
        missing = _phase_evaluate(
            topic, len(final), cfg["min_grounded_facts"],
            user_prompt=video_goal,
            must_cover=must_cover_list,
        )
        if not missing:
            break
        extra_raw = _phase_reflect_crawl(
            topic, missing, language, {s.url for s in sources}, emit, extraction_hint=extraction_hint
        )
        if not extra_raw:
            break
        new_sources = _classify_pages_by_tier(extra_raw, aliases)
        for s in new_sources:
            _chunks_with_enrichment(s, cfg["chunk_size_chars"], cfg["chunk_min_chars"], cfg["chunk_overlap_chars"])
        # Batch all gap-fill chunks into a single LLM call to stay under RPM caps.
        enriched_gap: list[dict] = []
        gap_idx = 0
        for s in new_sources:
            for ch in s.chunks:
                window_text = f"{ch.preceding_context}\n\n{ch.text}\n\n{ch.following_context}"
                enriched_gap.append({
                    "index": gap_idx,
                    "source_url": s.url,
                    "source_domain": _domain_of(s.url),
                    "page_title": s.title,
                    "section_heading": ch.section_heading,
                    "authority_tier": int(s.authority_tier or 4),
                    "window_text": window_text,
                })
                gap_idx += 1
        new_raw_facts = _extract_batch_one_call(
            enriched_gap, topic, aliases, extraction_hint,
            video_goal=video_goal, must_cover=must_cover_str,
        ) if enriched_gap else []
        if forbidden:
            new_raw_facts = [rf for rf in new_raw_facts if not is_contaminated(rf["claim"], forbidden)[0]]
        new_grounded, _ = _verify_grounding(new_raw_facts, aliases, cfg["grounding_fuzzy_threshold"])
        final = _score_and_rank_facts(final + new_grounded, cfg["max_final_facts"])
        sources.extend(new_sources)
        fact_ok, diversity_ok = _check_quality_floor(final, cfg, warnings, must_cover_list=must_cover_list)
        coverage.min_fact_threshold_met = fact_ok
        coverage.min_tier_diversity_met = diversity_ok
        iteration += 1

    # --- Stage 9: Final gate ------------------------------------------
    if not fact_ok:
        warnings.append(
            f"Proceeding with {len(final)} grounded facts (below floor of "
            f"{cfg['min_grounded_facts']}). Video quality will be degraded."
        )

    # Persist
    try:
        _store_to_bank(topic, [f.to_legacy_dict() for f in final], language)
    except Exception:
        pass

    gc.collect()

    return _build_legacy({
        "facts": final,
        "coverage": coverage,
        "warnings": warnings,
        "elapsed": time.monotonic() - start,
        "cache_hit": False,
        "sources": sources,
    })
