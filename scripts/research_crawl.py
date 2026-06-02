"""
Crawl URLs for a topic and index them in the local RAG store.

Usage:
    .venv/bin/python scripts/research_crawl.py "<topic>" <url1> <url2> ...

Reuses src/agent/research_agent.py primitives (Wikipedia fetch, BM25 crawl,
tier classification, chunking) and src/agent/rag_store.py (ChromaDB + BM25
+ rerank), but skips ALL LLM calls. Output is one-line JSON for the caller
(typically the /video skill) to read.

Re-running with the same topic within TTL (default 7 days) is a no-op:
prints {"cache_hit": true, ...} and exits 0 without crawling.
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main():
    if len(sys.argv) < 3:
        print('Usage: python scripts/research_crawl.py "<topic>" <url1> [url2 ...]', file=sys.stderr)
        sys.exit(2)

    topic = sys.argv[1].strip()
    urls = [u.strip() for u in sys.argv[2:] if u.strip()]
    if not topic or not urls:
        print("Error: topic and at least one URL required", file=sys.stderr)
        sys.exit(2)

    from src.agent.rag_store import RagStore
    from src.agent.research_agent import (
        _chunks_with_enrichment,
        _classify_pages_by_tier,
        _crawl_with_soft_floor,
        _phase_wiki,
        _prepare_crawl_urls,
        _SKIP_DOMAINS,
    )
    from src.agent_config import research_settings

    cfg = research_settings()
    rag = RagStore(topic, ttl_secs=cfg["rag_cache_ttl_secs"])

    if rag.is_cached():
        print(json.dumps({
            "topic": topic,
            "cache_hit": True,
            "chunks_indexed": rag.count(),
            "sources": [],
        }, ensure_ascii=False))
        return

    pseudo_results = [{"url": u} for u in urls]
    crawl_urls = _prepare_crawl_urls(pseudo_results, _SKIP_DOMAINS)

    pages: list[dict] = []
    if crawl_urls:
        try:
            pages = _crawl_with_soft_floor(
                crawl_urls[:cfg["max_crawl_pages"]],
                query=topic,
                base_threshold=cfg["bm25_threshold"],
            )
        except Exception as exc:
            print(f"Crawl error: {exc}", file=sys.stderr)
            pages = []

    try:
        wiki_pages = _phase_wiki(topic, "en-US", emit=None)
    except Exception:
        wiki_pages = []

    all_pages = list(wiki_pages) + list(pages)
    if not all_pages:
        print(json.dumps({
            "topic": topic,
            "cache_hit": False,
            "chunks_indexed": 0,
            "sources": [],
            "error": "no pages retrieved",
        }, ensure_ascii=False))
        sys.exit(1)

    sources = _classify_pages_by_tier(all_pages, [topic])
    chunk_dicts: list[dict] = []
    for s in sources:
        chunks = _chunks_with_enrichment(
            s,
            chunk_size=cfg["chunk_size_chars"],
            chunk_min=cfg["chunk_min_chars"],
            overlap=cfg["chunk_overlap_chars"],
        )
        for j, ch in enumerate(chunks):
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

    indexed = rag.add_chunks(chunk_dicts) if chunk_dicts else 0

    seen_urls: set[str] = set()
    unique_sources: list[dict] = []
    for s in sources:
        if s.url and s.url not in seen_urls:
            seen_urls.add(s.url)
            unique_sources.append({"url": s.url, "tier": s.authority_tier})

    print(json.dumps({
        "topic": topic,
        "cache_hit": False,
        "chunks_indexed": indexed,
        "sources": unique_sources,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
