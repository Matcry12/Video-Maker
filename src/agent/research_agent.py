"""Research Agent — multi-phase RAG pipeline for finding viral facts.

Pipeline: multi-query search → wiki → crawl → chunk → dedup →
          BM25 RAG index → retrieve → LLM rerank → return facts.

All search/fetch/extract functions are reused from content_sources/.
"""

import gc
import json
import logging
import os
import re
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_DDG_PER_QUERY = 12
_MAX_CRAWL_PAGES = 5
_MAX_SOURCE_UNITS = 60
_MAX_RETRIEVAL_RESULTS = 20
_MAX_FINAL_FACTS = 12

_QUERY_TEMPLATES = [
    "{topic} hidden details",
    "{topic} dark facts",
    "{topic} things you didn't know",
    "{topic} controversy",
    "{topic} behind the scenes",
    "{topic} fan theory",
    "{topic} easter eggs secrets",
    "{topic} psychological analysis",
]

_ENGLISH_QUERY_TEMPLATES = [
    "{eng} hidden details things you didn't know",
    "{eng} dark facts secrets",
    "{eng} fan theory controversy",
    "{eng} creator interview behind the scenes",
    "{eng} easter eggs cut content",
]

_RETRIEVAL_TEMPLATES = [
    "{topic} hidden secret detail nobody knows",
    "{topic} shocking number death toll statistic",
    "{topic} creator intention behind the scenes reason",
    "{topic} dark theory controversy debate",
]


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
        clean = re.sub(r"[^A-Za-z0-9:'\-]", "", token)
        if clean and len(clean) >= 2 and all(ord(c) < 256 for c in clean):
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
    # Replace HTML tags with spaces (not just strip — preserves word boundaries)
    text = re.sub(r"<[^>]+>", " ", text)
    # DDG bold markers
    text = text.replace("\x02", " ").replace("\x03", " ")
    # Fix camelCase smashing: insert space before ASCII uppercase after ASCII lowercase
    # Only ASCII ranges to avoid breaking Vietnamese diacritics
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Fix letter-digit boundaries (ASCII only)
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _emit_progress(emit: Optional[Callable], phase: str, message: str):
    if emit:
        try:
            emit({"phase": phase, "message": message})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Phase 1: Multi-query search
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

    # Build diverse query set
    queries = list(search_queries) if search_queries else []

    # Expand with templates if fewer than 5
    if len(queries) < 5:
        for tmpl in _QUERY_TEMPLATES:
            q = tmpl.format(topic=topic)
            if q not in queries:
                queries.append(q)
            if len(queries) >= 8:
                break

    # For non-English: add English variants
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
            # Use en-US for English queries even when target language differs
            search_lang = language
            if query and all(ord(c) < 256 for c in query.replace(" ", "").replace("-", "").replace("'", "")):
                search_lang = "en-US"

            results = search_duckduckgo(query, language=search_lang, max_results=_MAX_DDG_PER_QUERY)
            for r in results:
                url = r.get("href", r.get("url", ""))
                snippet = r.get("body", r.get("snippet", ""))
                title = r.get("title", "")

                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)

                if snippet and len(snippet) > 50:
                    clean_snippet = _clean_html_text(snippet)
                    if len(clean_snippet) > 50:
                        snippet_chunks.append({
                            "text": clean_snippet[:500],
                            "title": title,
                            "source_url": url,
                            "source": "duckduckgo",
                        })
        except Exception as exc:
            logger.debug("DDG search failed for '%s': %s", query, exc)

    _emit_progress(emit, "research", f"Found {len(all_results)} unique URLs, {len(snippet_chunks)} snippets")
    return all_results, snippet_chunks


# ---------------------------------------------------------------------------
# Phase 2: Wikipedia
# ---------------------------------------------------------------------------

def _phase_wiki(
    topic: str,
    language: str,
    emit: Optional[Callable],
) -> list[dict]:
    """Fetch Wikipedia in target language + English. Returns section chunks."""
    from ..content_sources.wikipedia_source import fetch_wikipedia_draft

    wiki_chunks: list[dict] = []

    # Target language Wikipedia
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
                    "source": "wikipedia",
                })
    except Exception as exc:
        logger.debug("Wikipedia fetch failed (%s): %s", language, exc)

    # English Wikipedia for non-English topics
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
                            "source": "wikipedia",
                        })
            except Exception as exc:
                logger.debug("English Wikipedia fetch failed: %s", exc)

    _emit_progress(emit, "research", f"Wikipedia: {len(wiki_chunks)} sections")
    return wiki_chunks


# ---------------------------------------------------------------------------
# Phase 3: Crawl promising URLs
# ---------------------------------------------------------------------------

def _phase_crawl(
    search_results: list[dict],
    emit: Optional[Callable],
) -> list[dict]:
    """Crawl top URLs from search results. Returns crawled text chunks."""
    if not search_results:
        return []

    # Score and filter URLs
    scored: list[tuple[float, dict]] = []
    for r in search_results:
        url = r.get("href", r.get("url", ""))
        if not url:
            continue
        # Skip Wikipedia (already fetched in phase 2)
        if "wikipedia.org" in url:
            continue

        score = 0.5
        url_lower = url.lower()
        # Prefer content-rich sites
        if "fandom.com" in url_lower or "wiki" in url_lower:
            score += 0.3
        if "screenrant.com" in url_lower or "cbr.com" in url_lower:
            score += 0.2
        if "reddit.com" in url_lower:
            score += 0.15
        if "youtube.com" in url_lower or "tiktok.com" in url_lower:
            score -= 0.3  # video sites have little text
        scored.append((score, r))

    scored.sort(key=lambda t: -t[0])
    top_results = [r for _s, r in scored[:_MAX_CRAWL_PAGES]]

    if not top_results:
        return []

    _emit_progress(emit, "research", f"Crawling {len(top_results)} pages...")

    crawl_chunks: list[dict] = []
    try:
        from ..content_sources.crawl4ai_source import crawl_search_results
        crawled = crawl_search_results(top_results, max_pages=_MAX_CRAWL_PAGES)
        for page in crawled:
            text = page.get("text", "")
            if text and len(text) > 100:
                crawl_chunks.append({
                    "text": text[:4000],
                    "title": page.get("title", ""),
                    "source_url": page.get("url", ""),
                    "source": "web",
                })
    except Exception as exc:
        logger.debug("Crawl4AI failed: %s", exc)
        # Fallback: simple HTTP fetch for top URLs
        for r in top_results[:3]:
            url = r.get("href", r.get("url", ""))
            if url:
                chunk = _simple_crawl(url)
                if chunk:
                    crawl_chunks.append(chunk)

    _emit_progress(emit, "research", f"Crawled {len(crawl_chunks)} pages")
    gc.collect()
    return crawl_chunks


def _simple_crawl(url: str) -> Optional[dict]:
    """Simple HTTP fetch as crawl fallback."""
    try:
        from urllib.request import Request, urlopen
        req = Request(url, headers={"User-Agent": "VideoMaker/1.0"})
        with urlopen(req, timeout=15) as resp:
            raw = resp.read().decode(
                resp.headers.get_content_charset() or "utf-8", errors="replace"
            )
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 200:
            return {
                "text": text[:4000],
                "title": "",
                "source_url": url,
                "source": "web",
            }
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Phase 4+5: Chunk, clean, deduplicate
# ---------------------------------------------------------------------------

def _phase_chunk_and_dedup(
    wiki_chunks: list[dict],
    crawl_chunks: list[dict],
    snippet_chunks: list[dict],
    topic: str,
    language: str,
    emit: Optional[Callable],
) -> list:
    """Merge all content, split into SourceUnits, compress, dedup.

    Returns list of SourceUnit objects (or dicts).
    """
    from ..content_sources.text_compressor import compress_for_llm

    _emit_progress(emit, "research", "Chunking and deduplicating content...")

    # Build a unified source_draft for extract_pipeline
    # Track original source per section so we can restore it after extraction
    all_sections: list[dict] = []
    section_source_map: dict[str, dict] = {}  # section_id -> {source, source_url}
    section_id = 1

    # Prioritize deep content (crawled + wiki) over shallow DDG snippets.
    # Crawled pages have 1000-4000 chars of real content vs snippets' 100-200 chars.
    # Cap snippets to prevent them from diluting crawled content in the pipeline.
    _MAX_SNIPPETS = 15
    capped_snippets = snippet_chunks[:_MAX_SNIPPETS]

    # Order matters: crawled first (best rank), wiki second, snippets last
    ordered_chunks = crawl_chunks + wiki_chunks + capped_snippets

    for chunk in ordered_chunks:
        text = chunk.get("text", "")
        if not text or len(text) < 30:
            continue
        # Compress each chunk to remove boilerplate
        compressed = compress_for_llm(text, max_chars=1500)
        if len(compressed) < 30:
            continue
        sid = f"s{section_id:03d}"
        all_sections.append({
            "section_id": sid,
            "title": chunk.get("title", ""),
            "text": compressed,
            "rank": section_id,
            "source_url": chunk.get("source_url", ""),
            "token_estimate": len(compressed.split()),
        })
        section_source_map[sid] = {
            "source": chunk.get("source", "web"),
            "source_url": chunk.get("source_url", ""),
        }
        section_id += 1

    if not all_sections:
        _emit_progress(emit, "research", "No content to chunk")
        return []

    source_draft = {
        "source": "multi_search",
        "topic_query": topic,
        "language": language,
        "sections": all_sections,
    }

    # Use existing extract pipeline: split → local score → Jaccard dedup → top-k
    from ..content_sources.extract_pipeline import extract_source_units_from_draft

    result = extract_source_units_from_draft(
        source_draft,
        max_chars_per_unit=500,
        min_chars_per_unit=80,
        dedupe_threshold=0.85,
        keep_top_k=_MAX_SOURCE_UNITS,
    )

    top_units = result.get("top_units", [])
    meta = result.get("meta", {})

    # Restore original source and URL from section_source_map
    for unit in top_units:
        doc_id = unit.get("doc_id", "")
        if doc_id in section_source_map:
            unit["source"] = section_source_map[doc_id]["source"]
            unit["source_url"] = section_source_map[doc_id]["source_url"]
            unit["url"] = section_source_map[doc_id]["source_url"]

    _emit_progress(
        emit, "research",
        f"Chunked: {meta.get('unit_count', '?')} units → "
        f"{meta.get('deduped_unit_count', '?')} after dedup → "
        f"{len(top_units)} selected"
    )
    return top_units


# ---------------------------------------------------------------------------
# Phase 6: Filter common knowledge
# ---------------------------------------------------------------------------

def _phase_filter_common(
    units: list[dict],
    topic: str,
    language: str,
    emit: Optional[Callable],
) -> list[dict]:
    """LLM filters out basic plot summaries and common knowledge."""
    if not units or len(units) <= 3:
        return units  # too few to filter

    _emit_progress(emit, "research", "Filtering common knowledge...")

    # Build text batch for LLM classification
    unit_texts = []
    for i, u in enumerate(units[:30]):  # cap at 30 to fit context
        text = u.get("text", "")[:300]
        unit_texts.append(f"[{i}] {text}")

    prompt_text = "\n".join(unit_texts)

    try:
        from ..llm_client import chat_completion

        system = (
            f"You are filtering content about '{topic}' for a viral YouTube Short.\n"
            "For each numbered chunk, reply with JUST the number and KEEP or REJECT.\n"
            "REJECT = basic plot summary, character introduction, common knowledge that any fan knows, "
            "generic biographical info, release dates, box office numbers.\n"
            "KEEP = hidden details, dark secrets, surprising numbers/statistics, fan theories, "
            "easter eggs, creator intentions, cut content, psychological analysis, controversies.\n"
            "When in doubt, KEEP.\n\n"
            "Example output:\n[0] KEEP\n[1] REJECT\n[2] KEEP\n..."
        )

        response = chat_completion(
            system=system,
            user=prompt_text,
            temperature=0.1,
            timeout=20.0,
        )

        # Parse response
        reject_indices: set[int] = set()
        for line in response.splitlines():
            m = re.match(r"\[(\d+)\]\s*(KEEP|REJECT)", line.strip(), re.IGNORECASE)
            if m and m.group(2).upper() == "REJECT":
                reject_indices.add(int(m.group(1)))

        filtered = [u for i, u in enumerate(units) if i not in reject_indices]
        rejected_count = len(units) - len(filtered)

        _emit_progress(emit, "research", f"Filtered: kept {len(filtered)}, rejected {rejected_count} common knowledge")

        # Safety: don't filter too aggressively
        if len(filtered) < 5 and len(units) >= 5:
            _emit_progress(emit, "research", "Filter too aggressive, keeping all")
            return units

        gc.collect()
        return filtered if filtered else units

    except Exception as exc:
        logger.debug("Common knowledge filter failed: %s", exc)
        _emit_progress(emit, "research", "Filter skipped (LLM unavailable)")
        return units


# ---------------------------------------------------------------------------
# Phase 7+8: RAG index + retrieve
# ---------------------------------------------------------------------------

def _phase_rag_retrieve(
    units: list[dict],
    topic: str,
    language: str,
    emit: Optional[Callable],
) -> list[dict]:
    """Hybrid BM25 + Dense retrieval with topic-entity relevance filter."""
    if not units:
        return []

    from ..content_sources.rag_index import (
        RAGIndex, DenseIndex, hybrid_retrieve, filter_by_topic_relevance,
    )

    _emit_progress(emit, "research", f"Building hybrid RAG index ({len(units)} chunks)...")

    # Build both indices
    bm25_index = RAGIndex()
    bm25_index.add_many(units)

    dense_index = DenseIndex()
    dense_index.add_many(units)

    # Generate retrieval queries
    retrieval_queries = _generate_retrieval_queries(topic, language)

    _emit_progress(emit, "research", f"Hybrid retrieving with {len(retrieval_queries)} queries...")

    # Hybrid retrieve and merge
    seen_ids: set[str] = set()
    merged: list[tuple[dict, float]] = []

    for q in retrieval_queries:
        results = hybrid_retrieve(bm25_index, dense_index, q, top_k=10)
        for doc, score in results:
            uid = doc.get("unit_id", doc.get("text", "")[:50])
            if uid not in seen_ids:
                seen_ids.add(uid)
                merged.append((doc, score))

    # Cap results
    merged = merged[:_MAX_RETRIEVAL_RESULTS]

    # Topic-entity relevance filter
    _emit_progress(emit, "research", "Filtering by topic relevance...")
    filtered = filter_by_topic_relevance(merged, topic, penalty=0.3, min_relevant=4)

    # Extract docs only (drop scores)
    result_docs = [doc for doc, _score in filtered]

    _emit_progress(emit, "research", f"Retrieved {len(result_docs)} topic-relevant chunks")

    # Cleanup
    bm25_index.clear()
    dense_index.clear()
    gc.collect()

    return result_docs


def _generate_retrieval_queries(topic: str, language: str) -> list[str]:
    """Generate detailed retrieval queries via LLM or templates."""
    # Try LLM first
    try:
        from ..llm_client import chat_completion

        system = (
            "Generate exactly 4 specific retrieval queries to find the most "
            "HIDDEN and SURPRISING facts about a topic. Each query should target "
            "a different angle: dark secrets, shocking statistics, creator intentions, "
            "hidden symbolism, fan theories, cut content. "
            "Return JSON only: {\"queries\": [\"...\", \"...\", \"...\", \"...\"]}"
        )
        user = f"Topic: {topic}\nLanguage: {language}"

        response = chat_completion(
            system=system,
            user=user,
            temperature=0.3,
            timeout=15.0,
        )

        # Parse JSON
        m = re.search(r"\{.*\}", response, re.DOTALL)
        if m:
            data = json.loads(m.group())
            queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
            if len(queries) >= 3:
                return queries[:6]
    except Exception as exc:
        logger.debug("Retrieval query generation failed: %s", exc)

    # Template fallback
    english_topic = _extract_english_query(topic) or topic
    return [tmpl.format(topic=english_topic) for tmpl in _RETRIEVAL_TEMPLATES]


# ---------------------------------------------------------------------------
# Phase 9: Rerank
# ---------------------------------------------------------------------------

def _phase_rerank(
    chunks: list[dict],
    topic: str,
    language: str,
    emit: Optional[Callable],
) -> list[dict]:
    """Use interest_ranker to score and select best facts."""
    if not chunks:
        return []

    _emit_progress(emit, "research", f"Reranking {len(chunks)} chunks...")

    from ..content_sources.extract_pipeline import to_rank_candidates
    from ..content_sources.interest_ranker import rank_interest_candidates
    from ..content_sources.models import SourceUnit

    # Build lookup map: candidate_id -> original chunk
    chunk_by_id: dict[str, dict] = {}
    source_units = []
    for i, chunk in enumerate(chunks):
        uid = chunk.get("unit_id", f"r{i:03d}")
        chunk_by_id[uid] = chunk
        try:
            su = SourceUnit(
                unit_id=uid,
                source=chunk.get("source", "web"),
                topic_query=topic,
                doc_id=chunk.get("doc_id", f"d{i:03d}"),
                title=chunk.get("title", ""),
                url=chunk.get("source_url", chunk.get("url", "")),
                text=chunk.get("text", ""),
                language=language,
                position=i,
                local_score=float(chunk.get("local_score", 0.5)),
                local_signals=chunk.get("local_signals", {}),
            )
            source_units.append(su)
        except Exception:
            continue

    if not source_units:
        return chunks

    candidates = to_rank_candidates(source_units)

    model = (
        os.getenv("GROQ_RESEARCH_MODEL")
        or os.getenv("GROQ_MODEL")
        or "llama-3.3-70b-versatile"
    )

    try:
        rank_result = rank_interest_candidates(
            candidates,
            model=model,
            language=language,
            keep_threshold=0.45,
        )

        items = rank_result.get("items", [])
        # Filter and sort
        kept = [
            item for item in items
            if item.get("keep", False) and item.get("final_score", 0) >= 0.4
        ]
        kept.sort(key=lambda x: -x.get("final_score", 0))

        _emit_progress(emit, "research", f"Reranked: {len(kept)}/{len(items)} kept")

        if not kept:
            return sorted(chunks, key=lambda c: -float(c.get("local_score", 0)))[:_MAX_FINAL_FACTS]

        # Merge reranker output with original chunk data (source, url, text)
        merged = []
        for item in kept[:_MAX_FINAL_FACTS]:
            cid = item.get("candidate_id", "")
            original = chunk_by_id.get(cid, {})
            merged.append({
                "text": original.get("text", item.get("text", "")),
                "source": original.get("source", "web"),
                "source_url": original.get("source_url", original.get("url", "")),
                "url": original.get("source_url", original.get("url", "")),
                "title": original.get("title", ""),
                "final_score": item.get("final_score", 0),
                "interest_score": item.get("interest_score", 0),
                "local_score": item.get("local_score", 0),
                "hook": item.get("hook", ""),
                "reason_tags": item.get("reason_tags", []),
            })
        return merged

    except Exception as exc:
        logger.warning("Reranking failed: %s", exc)
        _emit_progress(emit, "research", "Reranking failed, using local scores")
        return sorted(chunks, key=lambda c: -float(c.get("local_score", 0)))[:_MAX_FINAL_FACTS]


# ---------------------------------------------------------------------------
# Format facts for script agent
# ---------------------------------------------------------------------------

def _format_facts(ranked_items: list[dict], language: str) -> list[dict]:
    """Convert ranked items to the fact format expected by script agent."""
    facts = []
    seen_text: set[str] = set()  # final dedup by content prefix

    for item in ranked_items:
        text = item.get("text", "")
        if not text:
            continue

        text = _strip_markdown(text)
        if len(text) < 20:
            continue

        # Skip near-duplicate facts in final output
        prefix = text[:60].lower()
        if prefix in seen_text:
            continue
        seen_text.add(prefix)

        # Generate hook_text from first sentence if missing
        hook = item.get("hook", "").strip()
        if not hook:
            # Take first sentence (up to first period/question mark) as hook
            for sep in [". ", "! ", "? ", ".\n"]:
                if sep in text:
                    hook = text[:text.index(sep) + 1].strip()
                    break
            if not hook:
                hook = text[:100].strip()

        facts.append({
            "fact_text": text,
            "source": item.get("source", "web"),
            "score": float(item.get("final_score", item.get("interest_score", item.get("local_score", 0.5)))),
            "hook_text": hook,
            "source_url": item.get("url", item.get("source_url", "")),
            "reason_tags": item.get("reason_tags", []) or [],
        })

    return facts


# ---------------------------------------------------------------------------
# Last-resort LLM fallback
# ---------------------------------------------------------------------------

def _last_resort_llm(topic: str, language: str) -> list[dict]:
    """When all search fails, ask LLM directly. Last resort."""
    try:
        from ..llm_client import chat_completion

        if language.startswith("vi"):
            prompt = (
                f"Hãy cho tôi 6 sự thật THÚ VỊ, BẤT NGỜ và CỤ THỂ về: {topic}. "
                f"Chỉ viết những chi tiết ẨN GIẤU mà đa số fan không biết. "
                f"Bao gồm chi tiết như tên, số liệu, ngày tháng. Viết bằng tiếng Việt. "
                f"Không dùng markdown, không dùng **bold**, không dùng ### heading."
            )
        else:
            prompt = (
                f"Tell me 6 HIDDEN, SURPRISING facts about: {topic}. "
                f"Only include details most fans don't know. "
                f"Include names, numbers, dates. No markdown formatting."
            )

        answer = chat_completion(
            system="You are a research assistant. Give specific, surprising facts only.",
            user=prompt,
            temperature=0.4,
            timeout=20.0,
        )

        if not answer or len(answer) < 50:
            return []

        # Split into individual facts
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
# Main entry point
# ---------------------------------------------------------------------------

def run_research(
    topic: str,
    search_queries: list[str],
    language: str = "en-US",
    max_iterations: int = 5,
    timeout_sec: float = 180.0,
    emit: Optional[Callable[[dict], None]] = None,
) -> dict[str, Any]:
    """3-stage research pipeline: BM25 crawl → LLM extraction → dedup/format.

    Returns {"facts": [...], "sources_used": [...], "search_context": str, "warnings": [...]}
    """
    warnings: list[str] = []
    sources_used: set[str] = set()
    start_time = time.monotonic()

    _emit_progress(emit, "research", f"Researching '{topic}' (3-stage pipeline)...")

    # === STAGE 1: Search + BM25 Smart Crawl ===
    _emit_progress(emit, "research", "Stage 1: Searching and crawling...")

    # 1a. DDG search for URLs
    search_results, snippet_chunks = _phase_search(topic, search_queries, language, emit)
    if snippet_chunks:
        sources_used.add("duckduckgo")

    # 1b. Wikipedia (still useful for structured content)
    wiki_chunks = _phase_wiki(topic, language, emit)
    if wiki_chunks:
        sources_used.add("wikipedia")

    # 1c. BM25 smart crawl — topic-focused fit_markdown
    crawl_query = _extract_english_query(topic) or topic
    # Filter URLs — skip social media, video sites, and low-content sites
    _SKIP_DOMAINS = {"youtube.com", "tiktok.com", "pinterest.com", "instagram.com",
                     "twitter.com", "x.com", "facebook.com", "reddit.com/gallery",
                     "wikipedia.org", "amazon.com", "ebay.com"}
    crawl_urls = []
    seen_urls: set[str] = set()
    for r in search_results:
        url = r.get("href", r.get("url", ""))
        if not url or url in seen_urls:
            continue
        if any(domain in url.lower() for domain in _SKIP_DOMAINS):
            continue
        seen_urls.add(url)
        crawl_urls.append(url)

    crawled_pages: list[dict] = []
    if crawl_urls:
        _emit_progress(emit, "research", f"BM25 crawling {min(len(crawl_urls), _MAX_CRAWL_PAGES)} pages...")
        try:
            from ..content_sources.crawl4ai_source import crawl_with_bm25
            crawled_pages = crawl_with_bm25(
                crawl_urls[:_MAX_CRAWL_PAGES],
                query=crawl_query,
                bm25_threshold=1.5,
            )
            if crawled_pages:
                sources_used.add("web")
        except Exception as exc:
            logger.warning("BM25 crawl failed: %s", exc)
            warnings.append(f"BM25 crawl failed: {exc}")

    gc.collect()

    # Merge wiki as ONE combined "page" (not per-section — saves LLM calls)
    all_pages = list(crawled_pages)
    if wiki_chunks:
        combined_wiki = "\n\n".join(
            wc["text"] for wc in wiki_chunks if wc.get("text") and len(wc["text"]) > 50
        )
        if len(combined_wiki) > 200:
            all_pages.append({
                "url": wiki_chunks[0].get("source_url", ""),
                "title": "Wikipedia",
                "text": combined_wiki[:3000],
                "word_count": len(combined_wiki.split()),
            })

    # Cap total pages to extract — more than 6 wastes LLM calls
    _MAX_EXTRACT_PAGES = 6
    all_pages = all_pages[:_MAX_EXTRACT_PAGES]

    _emit_progress(emit, "research", f"Stage 1 done: {len(all_pages)} pages to extract from")

    if not all_pages:
        _emit_progress(emit, "research", "No content found — trying LLM fallback...")
        facts = _last_resort_llm(topic, language)
        if facts:
            sources_used.add("llm")
        return _build_result(facts, sources_used, "LLM fallback (no content)", warnings)

    # === STAGE 2: LLM Extraction per page (Gemma 12B, concurrent) ===
    _emit_progress(emit, "research", f"Stage 2: Extracting facts from {len(all_pages)} pages...")

    extracted_facts = _stage_extract(all_pages, topic, language, emit)

    _emit_progress(emit, "research", f"Stage 2 done: {len(extracted_facts)} raw facts extracted")

    if not extracted_facts:
        _emit_progress(emit, "research", "Extraction produced no facts — trying LLM fallback...")
        facts = _last_resort_llm(topic, language)
        if facts:
            sources_used.add("llm")
            warnings.append("LLM extraction returned 0 facts, used LLM fallback")
        return _build_result(facts, sources_used, "LLM fallback (extraction empty)", warnings)

    # === STAGE 3: Dedup + Format ===
    _emit_progress(emit, "research", "Stage 3: Deduplicating and formatting...")

    facts = _stage_dedup_and_format(extracted_facts, language)

    # Cap at max
    facts = facts[:_MAX_FINAL_FACTS]

    elapsed = time.monotonic() - start_time
    _emit_progress(emit, "research", f"Research complete — {len(facts)} facts in {elapsed:.1f}s")

    gc.collect()
    return _build_result(
        facts, sources_used,
        f"3-stage pipeline completed in {elapsed:.1f}s",
        warnings,
    )


# ---------------------------------------------------------------------------
# Stage 2: LLM extraction (Gemma 12B, concurrent)
# ---------------------------------------------------------------------------

_EXTRACT_SEMAPHORE_LIMIT = 1  # Sequential — avoids 503 rate limits on free tier


def _stage_extract(
    pages: list[dict],
    topic: str,
    language: str,
    emit: Optional[Callable],
) -> list[dict]:
    """Extract facts from each page using Gemma 12B. Runs concurrently."""
    from ..llm_client import extraction_completion

    system_prompt = (
        f"You are extracting HIDDEN and SURPRISING facts about: {topic}\n\n"
        f"RULES:\n"
        f"- Extract ONLY facts specifically about {topic}.\n"
        f"- REJECT basic info: character introductions, grade/rank, school affiliation, "
        f"generic personality descriptions. The audience ALREADY knows these.\n"
        f"- KEEP: hidden mechanics, creator insights, fan theories, dark secrets, "
        f"surprising numbers, cut content, behind-the-scenes decisions, unique techniques.\n"
        f"- Each fact must make a fan think 'I didn't know that!'\n"
        f"- NEVER invent or guess facts. Extract ONLY what is explicitly stated in the text.\n"
        f"- If a detail is not clearly stated in the content, do NOT include it.\n"
        f"- IGNORE facts about other characters unless they DIRECTLY involve {topic}.\n"
        f"- Return 3-6 high-quality facts per page. Quality over quantity.\n"
        f"- If the page has no hidden/surprising facts about {topic}, return empty.\n"
        f"- Return valid JSON only:\n"
        f'{{"is_relevant": true, "facts": ["fact 1", "fact 2", ...]}}\n'
        f'If not relevant: {{"is_relevant": false, "facts": []}}'
    )

    import concurrent.futures
    import threading

    sem = threading.Semaphore(_EXTRACT_SEMAPHORE_LIMIT)
    all_facts: list[dict] = []
    lock = threading.Lock()

    def _extract_one(page: dict) -> list[dict]:
        from ..content_sources.text_compressor import compress_for_llm

        # Pre-clean: compress to remove boilerplate, cap for 4B context
        raw_text = page.get("text", "")
        page_text = compress_for_llm(raw_text, max_chars=1500)
        if len(page_text) < 50:
            return []
        page_url = page.get("url", "")
        page_title = page.get("title", "")

        user_prompt = f"SOURCE: {page_url}\nTITLE: {page_title}\n\nCONTENT:\n{page_text}"

        sem.acquire()
        try:
            response = extraction_completion(
                system=system_prompt,
                user=user_prompt,
                temperature=0.1,
            )
        finally:
            sem.release()

        # Parse JSON response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            return []

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return []

        if not data.get("is_relevant", False):
            return []

        facts = []
        for fact_text in data.get("facts", []):
            fact_text = str(fact_text).strip()
            if len(fact_text) > 20:
                facts.append({
                    "text": _strip_markdown(fact_text),
                    "source": "web" if page_url != "search_snippets" else "duckduckgo",
                    "source_url": page_url,
                    "title": page_title,
                })
        return facts

    # Run extractions concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=_EXTRACT_SEMAPHORE_LIMIT) as executor:
        futures = {executor.submit(_extract_one, page): page for page in pages}
        for future in concurrent.futures.as_completed(futures):
            page = futures[future]
            try:
                page_facts = future.result()
                if page_facts:
                    with lock:
                        all_facts.extend(page_facts)
                    logger.info(
                        "Extracted %d facts from %s",
                        len(page_facts), page.get("url", "?")[:60],
                    )
            except Exception as exc:
                logger.warning("Extraction failed for %s: %s", page.get("url", "?")[:40], exc)

    return all_facts


# ---------------------------------------------------------------------------
# Stage 3: Dedup + format
# ---------------------------------------------------------------------------

def _stage_dedup_and_format(
    extracted_facts: list[dict],
    language: str,
) -> list[dict]:
    """Deduplicate extracted facts and format for script agent."""
    seen_prefixes: set[str] = set()
    unique: list[dict] = []

    for item in extracted_facts:
        text = item.get("text", "")
        if not text or len(text) < 20:
            continue

        # Prefix dedup
        prefix = text[:60].lower()
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)

        # Generate hook from first sentence
        hook = ""
        for sep in [". ", "! ", "? ", ".\n"]:
            if sep in text:
                hook = text[:text.index(sep) + 1].strip()
                break
        if not hook:
            hook = text[:100].strip()

        unique.append({
            "fact_text": text,
            "source": item.get("source", "web"),
            "score": 0.7,  # LLM-extracted facts get uniform high score
            "hook_text": hook,
            "source_url": item.get("source_url", ""),
            "reason_tags": ["extracted"],
        })

    return unique


def _build_result(
    facts: list[dict],
    sources_used: set[str],
    search_context: str,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "facts": facts,
        "sources_used": list(sources_used),
        "search_context": search_context,
        "warnings": warnings,
    }
