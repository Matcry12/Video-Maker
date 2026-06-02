"""
Query the local RAG store for a topic and print top chunks.

Usage:
    .venv/bin/python scripts/research_query.py "<topic>" "<query>" [--k 5] [--format md|json]

Default format is markdown (compact, token-efficient for LLM consumption).
Run `research_crawl.py` first to populate the topic's collection.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _format_markdown(chunks: list[dict]) -> str:
    if not chunks:
        return "_no results_"
    lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        url = c.get("source_url", "")
        tier = c.get("authority_tier", "?")
        title = c.get("page_title") or ""
        heading = c.get("section_heading") or ""
        text = (c.get("text") or "").strip()
        head = f"### [{i}] tier={tier} — {title}".rstrip(" -—")
        if heading:
            head += f" — {heading}"
        lines.append(head)
        if url:
            lines.append(f"<{url}>")
        lines.append("")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic")
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--format", choices=["md", "json"], default="md")
    ap.add_argument("--min-tier", type=int, default=None)
    args = ap.parse_args()

    from src.agent.rag_store import RagStore
    from src.agent_config import research_settings

    cfg = research_settings()
    rag = RagStore(args.topic, ttl_secs=cfg["rag_cache_ttl_secs"])

    if rag.count() == 0:
        print(f"No indexed chunks for topic {args.topic!r}. Run research_crawl.py first.", file=sys.stderr)
        sys.exit(1)

    chunks = rag.retrieve(
        [args.query],
        top_k=args.k,
        min_tier=args.min_tier,
        rrf_k=cfg["rag_cache_rrf_k"],
        tier_weights={
            1: cfg["tier_1_weight"],
            2: cfg["tier_2_weight"],
            3: cfg["tier_3_weight"],
            4: cfg["tier_4_weight"],
        },
    )

    if args.format == "json":
        slim = [{
            "text": c.get("text", ""),
            "source_url": c.get("source_url", ""),
            "page_title": c.get("page_title", ""),
            "authority_tier": c.get("authority_tier", 4),
            "section_heading": c.get("section_heading", ""),
        } for c in chunks]
        print(json.dumps({"topic": args.topic, "query": args.query, "chunks": slim}, ensure_ascii=False))
    else:
        print(_format_markdown(chunks))


if __name__ == "__main__":
    main()
