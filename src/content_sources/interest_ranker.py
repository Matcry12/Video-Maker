"""Compact candidate ranking for short-video interest extraction.

This module is designed for small LLMs. It expects already-cleaned,
already-segmented candidate units and keeps the model task narrow:
keep/reject, interest score, tags, and a short hook.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from hashlib import sha256
from pathlib import Path as _Path
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

_PROMPT_DIR = _Path(__file__).parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_KEEP_THRESHOLD = 0.62
MAX_INPUT_CHARS_PER_CANDIDATE = 500
MAX_HOOK_CHARS = 180
MAX_REASON_TAGS = 4
MAX_BATCH_SIZE = 14
MODEL_E2B_HINT = "e2b"
MODEL_E4B_HINT = "e4b"
ALLOWED_REASON_TAGS = {
    "surprising",
    "rare",
    "high_stakes",
    "counterintuitive",
    "dramatic",
    "visual",
    "historic",
    "emotionally_strong",
    "concrete",
    "weak",
}
PROMPT_TAG_LIST = ", ".join(sorted(ALLOWED_REASON_TAGS))
JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class RankCandidate(BaseModel):
    candidate_id: str
    text: str
    title: str = ""
    url: str = ""
    source: str
    topic_query: str
    language: str = "en-US"
    local_score: float = 0.0
    local_signals: dict[str, Any] = Field(default_factory=dict)


class RankResultItem(BaseModel):
    candidate_id: str
    keep: bool
    interest_score: float
    reason_tags: list[str] = Field(default_factory=list)
    hook: str = ""
    model_name: str = ""
    prompt_version: str = ""
    raw_status: str = "ok"
    parse_notes: list[str] = Field(default_factory=list)
    final_score: float = 0.0
    local_score: float = 0.0


def rank_interest_candidates(
    candidates: list[RankCandidate | dict[str, Any]],
    *,
    language: str,
    model: str = "",
    stage: str = "interest_rank",
    provider: str | None = None,
    batch_size: int | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    max_input_chars_per_candidate: int = MAX_INPUT_CHARS_PER_CANDIDATE,
    retry_on_parse_failure: bool = True,
    keep_threshold: float = DEFAULT_KEEP_THRESHOLD,
) -> dict[str, Any]:
    """Rank compact candidates with a small LLM and local fallbacks.

    If `model` is empty, resolves the Groq model from `stage` via agent_config.
    Ollama is auto-detected when `model` contains a ':'.
    """
    if not model:
        from ..agent_config import resolve_stage
        model = resolve_stage(stage).groq_model
    started_at = time.monotonic()
    normalized = _normalize_candidates(
        candidates,
        max_input_chars_per_candidate=max_input_chars_per_candidate,
    )
    effective_batch_size = _resolve_batch_size(model, requested=batch_size)
    batches = _batch_candidates(normalized, batch_size=effective_batch_size)

    items: list[RankResultItem] = []
    warnings: list[str] = []
    parsed_batches = 0
    repaired_batches = 0
    failed_batches = 0
    model_client: Any = None
    effective_provider = _resolve_provider(provider=provider, model=model)

    for batch in batches:
        batch_ids = [candidate.candidate_id for candidate in batch]
        batch_started_at = time.monotonic()
        parse_notes: list[str] = []
        batch_items: list[RankResultItem] | None = None
        repair_used = False

        try:
            model_client = model_client or _get_model_client(provider=effective_provider)
            prompt = _build_rank_prompt(
                batch,
                language=language,
                prompt_version=prompt_version,
            )
            raw_response = _call_model(
                provider=effective_provider,
                client=model_client,
                model=model,
                prompt=prompt,
                stage=stage,
            )
            batch_items, parse_notes = _parse_rank_response(
                raw_response,
                batch=batch,
                model_name=model,
                prompt_version=prompt_version,
            )
            parsed_batches += 1
        except Exception as exc:
            parse_notes.append(str(exc).strip() or type(exc).__name__)
            if retry_on_parse_failure:
                repair_used = True
                try:
                    repaired_response = _repair_rank_response(
                        provider=effective_provider,
                        client=model_client or _get_model_client(provider=effective_provider),
                        model=model,
                        invalid_output=locals().get("raw_response", ""),
                        stage=stage,
                    )
                    batch_items, repair_notes = _parse_rank_response(
                        repaired_response,
                        batch=batch,
                        model_name=model,
                        prompt_version=prompt_version,
                    )
                    parse_notes.extend(repair_notes)
                    parsed_batches += 1
                    repaired_batches += 1
                except Exception as repair_exc:
                    parse_notes.append(
                        f"repair_failed:{str(repair_exc).strip() or type(repair_exc).__name__}"
                    )

        if batch_items is None:
            failed_batches += 1
            warnings.append(
                f"Ranking batch failed for candidates {', '.join(batch_ids)}; used local fallback."
            )
            batch_items = _fallback_rank_items(
                batch,
                keep_threshold=keep_threshold,
                model_name=model,
                prompt_version=prompt_version,
                notes=parse_notes,
            )
        else:
            batch_items = _merge_rank_scores(batch, batch_items)
            if repair_used:
                for item in batch_items:
                    item.raw_status = "repaired"
                    if "repair_used" not in item.parse_notes:
                        item.parse_notes.append("repair_used")

        batch_latency_ms = round((time.monotonic() - batch_started_at) * 1000, 1)
        logger.info(
            "Interest rank batch complete model=%s size=%d parsed=%s repaired=%s latency_ms=%s ids=%s",
            model,
            len(batch),
            batch_items[0].raw_status if batch_items else "unknown",
            "yes" if repair_used else "no",
            batch_latency_ms,
            ",".join(batch_ids),
        )
        items.extend(batch_items)

    elapsed_ms = round((time.monotonic() - started_at) * 1000, 1)
    return {
        "items": [_model_dump(item) for item in items],
        "meta": {
            "model": model,
            "provider": effective_provider,
            "prompt_version": prompt_version,
            "batch_count": len(batches),
            "candidate_count": len(normalized),
            "parsed_batches": parsed_batches,
            "repaired_batches": repaired_batches,
            "failed_batches": failed_batches,
            "elapsed_ms": elapsed_ms,
            "batch_size": effective_batch_size,
            "cache_key": _build_cache_key(
                normalized,
                model=model,
                language=language,
                prompt_version=prompt_version,
            ),
        },
        "warnings": warnings,
    }


def _normalize_candidates(
    candidates: list[RankCandidate | dict[str, Any]],
    *,
    max_input_chars_per_candidate: int,
) -> list[RankCandidate]:
    normalized: list[RankCandidate] = []
    seen_ids: set[str] = set()

    for raw in candidates or []:
        candidate = raw if isinstance(raw, RankCandidate) else RankCandidate(**raw)
        candidate_id = str(candidate.candidate_id or "").strip()
        if not candidate_id or candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)

        cleaned_text = _trim_text_to_chars(
            _clean_text(candidate.text),
            max_input_chars_per_candidate,
        )
        if not cleaned_text:
            continue

        normalized.append(
            RankCandidate(
                candidate_id=candidate_id,
                text=cleaned_text,
                title=_clean_text(candidate.title),
                url=str(candidate.url or "").strip(),
                source=str(candidate.source or "").strip(),
                topic_query=_clean_text(candidate.topic_query),
                language=str(candidate.language or "en-US").strip() or "en-US",
                local_score=_clamp01(candidate.local_score),
                local_signals=dict(candidate.local_signals or {}),
            )
        )
    return normalized


def _resolve_batch_size(model: str, *, requested: int | None) -> int:
    if requested is not None:
        return max(1, min(int(requested), MAX_BATCH_SIZE))

    model_name = str(model or "").strip().lower()
    if MODEL_E2B_HINT in model_name:
        return 8
    if MODEL_E4B_HINT in model_name:
        return 10
    return 8


def _batch_candidates(candidates: list[RankCandidate], *, batch_size: int) -> list[list[RankCandidate]]:
    if not candidates:
        return []
    return [candidates[idx: idx + batch_size] for idx in range(0, len(candidates), batch_size)]


def _build_rank_prompt(
    candidates: list[RankCandidate],
    *,
    language: str,
    prompt_version: str,
) -> str:
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        row = f"{index}. candidate_id={candidate.candidate_id} | text={candidate.text}"
        if candidate.title:
            row += f" | title={candidate.title}"
        rows.append(row)

    return (
        "You are ranking short-video fact candidates.\n"
        "Use only the provided text.\n"
        "Return JSON only with this shape:\n"
        '{"items":[{"candidate_id":"...","keep":true,"interest_score":0.0,"reason_tags":["..."],"hook":"..."}]}\n\n'
        "Rules:\n"
        "- Output exactly one item per candidate.\n"
        "- Do not skip candidates.\n"
        "- Do not add prose outside JSON.\n"
        "- interest_score must be between 0 and 1.\n"
        "- keep should be false for bland, generic, repetitive, or weak candidates.\n"
        f"- reason_tags must come only from this list: {PROMPT_TAG_LIST}\n"
        "- hook must be one short sentence or phrase.\n"
        "- hook must be concrete, vivid, and audience-facing, not abstract.\n"
        "- Prefer hooks built from the strongest fact in the candidate.\n"
        "- If the candidate contains a rare number, extreme claim, death toll, severity rank, or unusual consequence, include that in the hook.\n"
        "- Avoid vague hooks like 'theory', 'hypothesis', 'ranking', 'conditions', 'overview', or 'history' unless the source text itself is weak.\n"
        "- Good hook style: 'One of only two level-7 nuclear disasters.'\n"
        "- Bad hook style: 'Nuclear accident severity ranking.'\n"
        f"- Requested language: {language}\n"
        f"- Prompt version: {prompt_version}\n\n"
        "Candidates:\n"
        + "\n".join(rows)
    )


def _resolve_provider(*, provider: str | None, model: str) -> str:
    provider_name = str(provider or "").strip().lower()
    if provider_name in {"groq", "ollama"}:
        return provider_name
    if ":" in str(model or ""):
        return "ollama"
    return "groq"


def _get_model_client(*, provider: str) -> Any:
    if provider == "ollama":
        return {
            "base_url": os.getenv("OLLAMA_BASE_URL", "").strip() or DEFAULT_OLLAMA_BASE_URL,
        }

    # Return None for groq — chat_completion handles API key internally
    return None


def _call_model(*, provider: str, client: Any, model: str, prompt: str, stage: str) -> str:
    if provider == "ollama":
        return _call_ollama_model(client=client, model=model, prompt=prompt)

    from ..llm_client import chat_completion

    return chat_completion(
        system=_load_prompt("interest_rank.txt"),
        user=prompt,
        stage=stage,
        model=model,
        temperature=0.2,
        timeout=25.0,
    )


def _call_ollama_model(*, client: dict[str, str], model: str, prompt: str) -> str:
    base_url = str(client.get("base_url") or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    endpoint = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0.2,
        },
        "messages": [
            {
                "role": "system",
                "content": _load_prompt("interest_rank.txt"),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120.0) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            raw = response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"Ollama HTTP error {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Ollama network error: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON envelope: {exc}") from exc

    message = (payload.get("message") or {}) if isinstance(payload, dict) else {}
    content = str(message.get("content") or "").strip()
    if not content:
        raise RuntimeError("Ollama returned empty content.")
    return content


def _repair_rank_response(*, provider: str, client: Any, model: str, invalid_output: str, stage: str) -> str:
    prompt = (
        "Fix this into valid JSON only.\n"
        "Required shape:\n"
        '{"items":[{"candidate_id":"...","keep":true,"interest_score":0.0,"reason_tags":["..."],"hook":"..."}]}\n\n'
        "Invalid output:\n"
        f"{invalid_output}"
    )
    return _call_model(provider=provider, client=client, model=model, prompt=prompt, stage=stage)


def _parse_rank_response(
    raw_response: str,
    *,
    batch: list[RankCandidate],
    model_name: str,
    prompt_version: str,
) -> tuple[list[RankResultItem], list[str]]:
    payload, notes = _load_json_payload(raw_response)
    items_raw = payload.get("items")
    if not isinstance(items_raw, list):
        raise ValueError("Response payload must contain an items array.")

    known_ids = {candidate.candidate_id: candidate for candidate in batch}
    results_by_id: dict[str, RankResultItem] = {}

    for raw_item in items_raw:
        if not isinstance(raw_item, dict):
            continue
        candidate_id = str(raw_item.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id not in known_ids or candidate_id in results_by_id:
            continue
        normalized_item = _validate_rank_item(
            raw_item,
            candidate=known_ids[candidate_id],
            model_name=model_name,
            prompt_version=prompt_version,
            inherited_notes=list(notes),
        )
        results_by_id[candidate_id] = normalized_item

    ordered_results: list[RankResultItem] = []
    for candidate in batch:
        item = results_by_id.get(candidate.candidate_id)
        if item is None:
            ordered_results.append(
                RankResultItem(
                    candidate_id=candidate.candidate_id,
                    keep=False,
                    interest_score=candidate.local_score,
                    reason_tags=["weak"] if candidate.local_score < DEFAULT_KEEP_THRESHOLD else [],
                    hook="",
                    model_name=model_name,
                    prompt_version=prompt_version,
                    raw_status="missing_item",
                    parse_notes=list(notes) + ["missing_from_model_output"],
                    local_score=candidate.local_score,
                    final_score=candidate.local_score,
                )
            )
        else:
            ordered_results.append(item)
    return ordered_results, notes


def _load_json_payload(raw_response: str) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    try:
        payload = json.loads(raw_response)
        if isinstance(payload, dict):
            return payload, notes
    except json.JSONDecodeError:
        notes.append("strict_parse_failed")

    match = JSON_OBJECT_RE.search(raw_response or "")
    if not match:
        raise ValueError("Could not find JSON object in model response.")

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Substring JSON parse failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Parsed JSON root is not an object.")
    notes.append("substring_parse_used")
    return payload, notes


def _validate_rank_item(
    raw_item: dict[str, Any],
    *,
    candidate: RankCandidate,
    model_name: str,
    prompt_version: str,
    inherited_notes: list[str],
) -> RankResultItem:
    keep = bool(raw_item.get("keep", False))
    interest_score = _clamp01(raw_item.get("interest_score", candidate.local_score))
    reason_tags = _normalize_reason_tags(raw_item.get("reason_tags"))
    hook = _trim_text_to_chars(_clean_text(raw_item.get("hook")), MAX_HOOK_CHARS)

    return RankResultItem(
        candidate_id=candidate.candidate_id,
        keep=keep,
        interest_score=interest_score,
        reason_tags=reason_tags,
        hook=hook,
        model_name=model_name,
        prompt_version=prompt_version,
        raw_status="ok",
        parse_notes=inherited_notes,
        local_score=candidate.local_score,
        final_score=_merge_score(candidate.local_score, interest_score),
    )


def _fallback_rank_items(
    batch: list[RankCandidate],
    *,
    keep_threshold: float,
    model_name: str,
    prompt_version: str,
    notes: list[str],
) -> list[RankResultItem]:
    fallback_items: list[RankResultItem] = []
    for candidate in batch:
        score = candidate.local_score
        fallback_items.append(
            RankResultItem(
                candidate_id=candidate.candidate_id,
                keep=score >= keep_threshold,
                interest_score=score,
                reason_tags=[] if score >= keep_threshold else ["weak"],
                hook="",
                model_name=model_name,
                prompt_version=prompt_version,
                raw_status="fallback_local",
                parse_notes=list(notes),
                local_score=score,
                final_score=score,
            )
        )
    return fallback_items


def _merge_rank_scores(
    batch: list[RankCandidate],
    items: list[RankResultItem],
) -> list[RankResultItem]:
    local_by_id = {candidate.candidate_id: candidate.local_score for candidate in batch}
    merged: list[RankResultItem] = []
    for item in items:
        local_score = _clamp01(local_by_id.get(item.candidate_id, item.local_score))
        item.local_score = local_score
        item.final_score = _merge_score(local_score, item.interest_score)
        merged.append(item)
    return merged


def _merge_score(local_score: float, model_score: float) -> float:
    combined = _clamp01(local_score) * 0.55 + _clamp01(model_score) * 0.45
    return round(combined, 4)


def _normalize_reason_tags(raw_tags: Any) -> list[str]:
    if isinstance(raw_tags, str):
        tags = [raw_tags]
    elif isinstance(raw_tags, list):
        tags = [str(tag or "").strip().lower() for tag in raw_tags]
    else:
        tags = []

    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag not in ALLOWED_REASON_TAGS or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
        if len(normalized) >= MAX_REASON_TAGS:
            break
    return normalized


def _clean_text(value: Any) -> str:
    return WHITESPACE_RE.sub(" ", str(value or "")).strip()


def _trim_text_to_chars(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars + 1]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" ,;:") + "..."


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    return round(max(0.0, min(1.0, number)), 4)


def _build_cache_key(
    candidates: list[RankCandidate],
    *,
    model: str,
    language: str,
    prompt_version: str,
) -> str:
    payload = {
        "model": model,
        "language": language,
        "prompt_version": prompt_version,
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "text_hash": sha256(candidate.text.encode("utf-8")).hexdigest()[:16],
            }
            for candidate in candidates
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return sha256(raw.encode("utf-8")).hexdigest()[:24]


def _model_dump(value: BaseModel) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()
