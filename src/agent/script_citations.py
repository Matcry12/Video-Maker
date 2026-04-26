"""Extract citations from a freshly-written script and match them to facts.

The writer is instructed to emit `[F003]` at the end of each sentence that
rests on fact F003. This module parses those markers, produces a CitationMap,
and returns a cleaned script with markers removed from visible text but kept
in structured metadata so the quality gate can consume them.
"""
from __future__ import annotations

import re
from typing import Any

from src.agent.contracts import Citation, CitationMap

# Recognize both valid fact markers like [F001] and empty placeholders like [...]
_CITE_RE = re.compile(r"\[(F\d{3,4}|\.\.\.)(?:\s*,\s*(F\d{3,4}|\.\.\.))*\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ỹ])")


def extract_citations(script: dict, valid_fact_ids: set[str]) -> tuple[dict, CitationMap]:
    """Return (cleaned_script, citation_map).

    The cleaned script has citation markers removed from visible text but keeps
    a new field `block["citations"] = [{"sentence_idx": i, "fact_ids": [...]}]`.
    """
    blocks = list(script.get("blocks") or [])
    citations: list[Citation] = []
    uncited: list[tuple[int, int]] = []
    used_ids: set[str] = set()

    new_blocks = []
    for b_idx, block in enumerate(blocks):
        text = block.get("text") or ""
        if not text.strip():
            new_blocks.append(dict(block))
            continue

        sentences = _split_sentences(text)
        cleaned_sentences = []
        block_citations_meta = []

        for s_idx, sent in enumerate(sentences):
            found_ids = _scan_citations(sent, valid_fact_ids)
            # Remove all citation-like brackets including [...] from visible text
            clean_sent = _CITE_RE.sub("", sent).rstrip().rstrip(",")
            cleaned_sentences.append(clean_sent)
            if found_ids:
                for fid in found_ids:
                    citations.append(Citation(fact_id=fid, block_idx=b_idx, sentence_idx=s_idx))
                    used_ids.add(fid)
                block_citations_meta.append({"sentence_idx": s_idx, "fact_ids": found_ids})
            else:
                uncited.append((b_idx, s_idx))

        new_block = dict(block)
        new_block["text"] = " ".join(cleaned_sentences).strip()
        new_block["citations"] = block_citations_meta
        new_blocks.append(new_block)

    cleaned = dict(script)
    cleaned["blocks"] = new_blocks
    unused = sorted(valid_fact_ids - used_ids)
    return cleaned, CitationMap(
        citations=citations,
        unused_fact_ids=unused,
        uncited_sentences=uncited,
    )


def _split_sentences(text: str) -> list[str]:
    """Split on sentence terminators. Preserve markers so they stay with their sentence."""
    if not text.strip():
        return []
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _scan_citations(sentence: str, valid_ids: set[str]) -> list[str]:
    ids: list[str] = []
    for m in _CITE_RE.finditer(sentence):
        bracket_content = sentence[m.start():m.end()][1:-1]
        for token in re.split(r"[\s,]+", bracket_content):
            tok = token.strip().upper()
            if tok in valid_ids:
                ids.append(tok)
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out
