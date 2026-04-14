"""Content source adapters used by the web layer."""

from .wikipedia_source import fetch_wikipedia_draft
from .multi_source import fetch_topic_content
from .llm_refiner import refine_draft_with_groq
from .interest_ranker import rank_interest_candidates, RankCandidate, RankResultItem
from .extract_pipeline import extract_source_units_from_draft, to_rank_candidates
from .fact_script_writer import write_script_from_facts, RankedFact
from .script_lint import lint_script

def ranked_items_to_facts(
    ranked_items: list,
    candidates: list,
) -> list[dict]:
    """Join RankResultItems back with original candidate text to build fact dicts.

    ranked_items: list of RankResultItem (or dicts) from rank_interest_candidates
    candidates: list of RankCandidate (or dicts) that were passed to the ranker
    Returns: list of fact dicts suitable for write_script_from_facts
    """
    text_map: dict[str, dict] = {}
    for c in candidates:
        d = c.model_dump() if hasattr(c, "model_dump") else (c if isinstance(c, dict) else {})
        cid = d.get("candidate_id", "")
        if cid:
            text_map[cid] = d

    facts: list[dict] = []
    for item in ranked_items:
        d = item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else {})
        if not d.get("keep", False):
            continue
        cid = d.get("candidate_id", "")
        orig = text_map.get(cid, {})
        fact_text = orig.get("text", "") or d.get("hook", "")
        if not fact_text:
            continue
        facts.append({
            "fact_text": fact_text,
            "hook_text": str(d.get("hook") or ""),
            "source_url": str(orig.get("url") or ""),
            "score": float(d.get("final_score") or d.get("interest_score") or d.get("local_score") or 0),
            "suggested_role": "",
            "reason_tags": list(d.get("reason_tags") or []),
        })
    return facts


__all__ = [
    "fetch_wikipedia_draft",
    "refine_draft_with_groq",
    "rank_interest_candidates",
    "RankCandidate",
    "RankResultItem",
    "extract_source_units_from_draft",
    "to_rank_candidates",
    "write_script_from_facts",
    "RankedFact",
    "lint_script",
    "ranked_items_to_facts",
    "fetch_topic_content",
]
