"""LLM/heuristic fact extraction for Topic Pool content bank."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import FactCard, ScoreBreakdown, TopicBundle

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_FACT_PROMPT_VERSION = "facts_v1"

MIN_FACTS_PER_TOPIC = 3
MAX_FACTS_PER_TOPIC = 20
MIN_FACT_WORDS = 6
MIN_FACT_CHARS = 28

_SPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9À-ỹ]+", flags=re.UNICODE)


def extract_fact_cards(
    topic_bundle: TopicBundle,
    *,
    facts_per_topic_target: int = 8,
    prompt_override: str | None = None,
    timeout_sec: float = 35.0,
) -> tuple[list[FactCard], dict[str, Any], list[str]]:
    """Extract fact cards for one topic bundle via Groq or heuristic fallback."""
    target = _clamp_target(facts_per_topic_target)
    warnings: list[str] = []

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if api_key:
        try:
            raw_facts, meta = _extract_with_groq(
                topic_bundle,
                api_key=api_key,
                target=target,
                prompt_override=prompt_override,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            warnings.append(f"LLM extraction unavailable, used heuristic fallback: {exc}")
            raw_facts = _extract_heuristic(topic_bundle, target=target)
            meta = {
                "provider": "groq",
                "status": "fallback_heuristic",
                "error": str(exc),
                "prompt_mode": "override" if prompt_override else "default",
                "prompt_version": DEFAULT_FACT_PROMPT_VERSION,
                "target_facts": target,
            }
    else:
        warnings.append("GROQ_API_KEY missing; used heuristic fallback for fact extraction.")
        raw_facts = _extract_heuristic(topic_bundle, target=target)
        meta = {
            "provider": "heuristic",
            "status": "no_api_key",
            "prompt_mode": "heuristic",
            "prompt_version": "heuristic_v1",
            "target_facts": target,
        }

    cards = _raw_facts_to_cards(topic_bundle, raw_facts, target=target)
    if not cards:
        warnings.append("No fact cards extracted from source text.")
    return cards, meta, warnings


def _extract_with_groq(
    topic_bundle: TopicBundle,
    *,
    api_key: str,
    target: int,
    prompt_override: str | None,
    timeout_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = (os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL)
    prompt, prompt_mode = _build_fact_prompt(
        topic_bundle,
        target=target,
        prompt_override=prompt_override,
    )

    payload = {
        "model": model,
        "temperature": 0.3,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You extract high-value short-video fact cards from source text. "
                    "Use only provided text and return strict JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    response = _post_json(
        GROQ_API_URL,
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout_sec=timeout_sec,
    )

    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("Groq response missing choices.")

    content = (((choices[0] or {}).get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError("Groq returned empty content.")

    facts = _parse_facts_json(content)
    usage = response.get("usage") or {}
    meta = {
        "provider": "groq",
        "status": "ok",
        "model": model,
        "prompt_mode": prompt_mode,
        "prompt_version": DEFAULT_FACT_PROMPT_VERSION,
        "target_facts": target,
        "tokens_in": usage.get("prompt_tokens"),
        "tokens_out": usage.get("completion_tokens"),
    }
    return facts, meta


def _build_fact_prompt(
    topic_bundle: TopicBundle,
    *,
    target: int,
    prompt_override: str | None,
) -> tuple[str, str]:
    topic_name = topic_bundle.wiki_title or topic_bundle.topic_query
    source_text = (topic_bundle.extended_text or topic_bundle.summary_text or "").strip()
    source_text = source_text[:12000]

    override = str(prompt_override or "").strip()
    if override:
        prompt = (
            f"{override}\n\n"
            "Output requirements:\n"
            "1) Return strict JSON object: {\"facts\":[...]} only.\n"
            "2) Use only provided source text.\n"
            f"3) Extract around {target} distinct fact cards.\n"
            "4) Each fact: fact_text, hook_text, evidence_text, confidence (0-1), "
            "novelty (0-1), relevance (0-1), hook_strength (0-1), clarity (0-1).\n"
            "5) Keep fact_text concise for short-video narration.\n"
            f"Topic: {topic_name}\n"
            f"Source URL: {topic_bundle.wiki_url}\n"
            f"Source text:\n{source_text}"
        )
        return prompt, "override"

    prompt = (
        "Extract short-video fact cards from the provided Wikipedia text.\n"
        "Rules:\n"
        "1) Use only information from the source text.\n"
        "2) Return strict JSON object only with this shape: "
        '{"facts":[{"fact_text":"...","hook_text":"...","evidence_text":"...",'
        '"confidence":0.0,"novelty":0.0,"relevance":0.0,"hook_strength":0.0,"clarity":0.0}, ...]}.\n'
        f"3) Target count: {target} facts.\n"
        "4) fact_text should be concise and narration-ready (1-2 sentences).\n"
        "5) evidence_text should be brief and source-grounded.\n"
        f"Topic: {topic_name}\n"
        f"Source URL: {topic_bundle.wiki_url}\n"
        f"Source text:\n{source_text}"
    )
    return prompt, "default"


def _extract_heuristic(topic_bundle: TopicBundle, *, target: int) -> list[dict[str, Any]]:
    text = (topic_bundle.extended_text or topic_bundle.summary_text or "").strip()
    sentences = [_clean_sentence(part) for part in _SENTENCE_RE.split(text) if _clean_sentence(part)]

    if not sentences and text:
        sentences = [text]

    selected: list[dict[str, Any]] = []
    seen = set()
    for sentence in sentences:
        key = _normalized_key(sentence)
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "fact_text": sentence,
                "hook_text": f"Why it matters: {sentence[:120].rstrip()}".strip(),
                "evidence_text": sentence[:180].strip(),
                "confidence": 0.62,
                "novelty": 0.62,
                "relevance": 0.68,
                "hook_strength": 0.6,
                "clarity": 0.72,
            }
        )
        if len(selected) >= target:
            break

    return selected


def _raw_facts_to_cards(
    topic_bundle: TopicBundle,
    raw_facts: list[dict[str, Any]],
    *,
    target: int,
) -> list[FactCard]:
    cards: list[FactCard] = []
    seen_keys = set()

    topic_label = topic_bundle.wiki_title or topic_bundle.topic_query
    source_url = topic_bundle.wiki_url or ""
    language = topic_bundle.language or "en-US"

    for raw in raw_facts:
        fact_text = _clean_sentence(str(raw.get("fact_text") or ""))
        if not fact_text or _is_low_quality_fact(fact_text):
            continue

        key = _normalized_key(fact_text)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)

        hook_text = _clean_sentence(str(raw.get("hook_text") or ""))
        evidence_text = _clean_sentence(str(raw.get("evidence_text") or ""))

        breakdown = ScoreBreakdown(
            novelty=_clamp01(raw.get("novelty")),
            relevance=_clamp01(raw.get("relevance")),
            hook_strength=_clamp01(raw.get("hook_strength")),
            clarity=_clamp01(raw.get("clarity")),
            confidence=_clamp01(raw.get("confidence")),
        )

        fact_id = _make_fact_id(topic_bundle.topic_id, fact_text, len(cards))
        cards.append(
            FactCard(
                fact_id=fact_id,
                topic_id=topic_bundle.topic_id,
                topic_label=topic_label,
                fact_text=fact_text,
                hook_text=hook_text,
                evidence_text=evidence_text,
                source_url=source_url,
                language=language,
                score_breakdown=breakdown,
                score=0.0,
            )
        )
        if len(cards) >= target:
            break

    return cards


def _parse_facts_json(content: str) -> list[dict[str, Any]]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    if not text.startswith("{"):
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            text = text[first : last + 1]

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM fact output is not valid JSON.") from exc

    facts = payload.get("facts") if isinstance(payload, dict) else None
    if not isinstance(facts, list):
        raise RuntimeError("LLM fact output missing 'facts' array.")

    normalized: list[dict[str, Any]] = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        fact_text = _clean_sentence(str(item.get("fact_text") or ""))
        if not fact_text or _is_low_quality_fact(fact_text):
            continue
        normalized.append(
            {
                "fact_text": fact_text,
                "hook_text": _clean_sentence(str(item.get("hook_text") or "")),
                "evidence_text": _clean_sentence(str(item.get("evidence_text") or "")),
                "confidence": item.get("confidence"),
                "novelty": item.get("novelty"),
                "relevance": item.get("relevance"),
                "hook_strength": item.get("hook_strength"),
                "clarity": item.get("clarity"),
            }
        )

    if not normalized:
        raise RuntimeError("LLM fact output contains no usable facts.")
    return normalized


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout_sec: float = 35.0,
) -> dict[str, Any]:
    merged_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if headers:
        merged_headers.update(headers)

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=merged_headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_sec) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq API HTTP error {exc.code}: {detail[:300]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Groq API network error: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Groq API request timed out.") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Groq API returned invalid JSON.") from exc


def _make_fact_id(topic_id: str, fact_text: str, index: int) -> str:
    raw = f"{topic_id}|{_normalized_key(fact_text)}|{index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _normalized_key(text: str) -> str:
    cleaned = re.sub(r"[^\wÀ-ỹ]+", " ", str(text or "").lower(), flags=re.UNICODE)
    cleaned = _SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _clean_sentence(text: str) -> str:
    raw = _SPACE_RE.sub(" ", str(text or "").strip())
    if not raw:
        return ""
    return raw[:400].strip()


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(str(text or "")))


def _is_low_quality_fact(text: str) -> bool:
    cleaned = _clean_sentence(text)
    if not cleaned:
        return True
    if len(cleaned) < MIN_FACT_CHARS:
        return True
    if _word_count(cleaned) < MIN_FACT_WORDS:
        return True

    lowered = cleaned.casefold()
    low_value_markers = {
        "lorem ipsum",
        "n/a",
        "unknown",
        "not available",
        "tbd",
    }
    for marker in low_value_markers:
        if marker in lowered:
            return True
    return False


def _clamp_target(value: int) -> int:
    try:
        n = int(value)
    except Exception:
        n = 8
    return max(MIN_FACTS_PER_TOPIC, min(MAX_FACTS_PER_TOPIC, n))


def _clamp01(value: Any) -> float:
    try:
        n = float(value)
    except Exception:
        return 0.0
    if n < 0.0:
        return 0.0
    if n > 1.0:
        return 1.0
    return n
