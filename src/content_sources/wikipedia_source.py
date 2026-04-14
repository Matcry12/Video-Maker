"""
Wikipedia source adapter (source_1).

Fetches topic summaries and maps them into the existing script format:
{
  "language": "...",
  "blocks": [{"text": "..."}]
}
"""

from __future__ import annotations

import html
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_MAX_BLOCKS = 5
MIN_BLOCKS = 1
MAX_BLOCKS = 12
MIN_TARGET_BLOCKS = 6
DEFAULT_TIMEOUT_SEC = 12.0
MIN_EXTRACT_LEN = 80
CACHE_TTL_SEC = 24 * 60 * 60
HTTP_RETRY_COUNT = 2
HTTP_RETRY_BACKOFF_SEC = 0.45
MAX_SOURCE_SENTENCES = 220
MAX_SOURCE_CHARS = 60000
MAX_SOURCE_SECTIONS = 96

PROJECT_ROOT = Path(__file__).parent.parent.parent
WIKI_CACHE_DIR = PROJECT_ROOT / "tmp" / "cache" / "wiki"

WIKIPEDIA_USER_AGENT = "VideoMaker/1.0 (source_1; wikipedia)"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CITATION_RE = re.compile(r"\[\d+\]")
_SPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def resolve_wiki_domain(language_code: str | None) -> str:
    """Map UI language code to Wikipedia domain."""
    if not language_code:
        return "en.wikipedia.org"

    raw = str(language_code).strip().lower()
    if not raw:
        return "en.wikipedia.org"

    primary = re.split(r"[-_]", raw, maxsplit=1)[0]
    if primary == "vi":
        return "vi.wikipedia.org"
    if primary == "en":
        return "en.wikipedia.org"
    if re.fullmatch(r"[a-z]{2,3}", primary):
        return f"{primary}.wikipedia.org"
    return "en.wikipedia.org"


def search_page(topic: str, language_code: str, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> dict[str, Any]:
    """Search a topic and return candidates with one selected title."""
    normalized_topic = _normalize_topic(topic)
    if not normalized_topic:
        raise ValueError("Topic must not be empty.")

    domain = resolve_wiki_domain(language_code)
    endpoint = f"https://{domain}/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": normalized_topic,
        "srlimit": "8",
        "utf8": "1",
        "format": "json",
    }

    response = _http_get_json(endpoint, params=params, timeout_sec=timeout_sec)
    raw_candidates = ((response or {}).get("query") or {}).get("search") or []

    candidates = []
    for idx, item in enumerate(raw_candidates):
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        snippet = _strip_html(str(item.get("snippet") or ""))
        score = _candidate_score(normalized_topic, title, snippet)
        candidates.append(
            {
                "title": title,
                "snippet": snippet,
                "pageid": item.get("pageid"),
                "rank": idx + 1,
                "score": round(score, 3),
            }
        )

    if not candidates:
        return {
            "domain": domain,
            "topic": normalized_topic,
            "candidates": [],
            "selected": None,
        }

    candidates.sort(key=lambda item: (-item["score"], item["rank"]))
    selected = candidates[0]

    # Move obvious disambiguation pages behind better candidates when possible.
    if _is_likely_disambiguation(selected["title"], selected["snippet"]):
        for candidate in candidates[1:]:
            if not _is_likely_disambiguation(candidate["title"], candidate["snippet"]):
                selected = candidate
                break

    return {
        "domain": domain,
        "topic": normalized_topic,
        "candidates": candidates,
        "selected": selected,
    }


def fetch_summary(title: str, language_code: str, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> dict[str, Any]:
    """Fetch a concise page summary from Wikipedia REST API."""
    safe_title = str(title or "").strip()
    if not safe_title:
        raise ValueError("Title must not be empty.")

    domain = resolve_wiki_domain(language_code)
    encoded_title = quote(safe_title.replace(" ", "_"), safe="")
    endpoint = f"https://{domain}/api/rest_v1/page/summary/{encoded_title}"
    response = _http_get_json(endpoint, timeout_sec=timeout_sec)

    result_title = str(response.get("title") or safe_title).strip()
    extract = _clean_extract_text(str(response.get("extract") or ""))
    page_type = str(response.get("type") or "").strip().lower()

    canonical_url = (
        (((response.get("content_urls") or {}).get("desktop") or {}).get("page"))
        or f"https://{domain}/wiki/{quote(result_title.replace(' ', '_'), safe='')}"
    )

    return {
        "domain": domain,
        "title": result_title,
        "extract": extract,
        "type": page_type,
        "canonical_url": canonical_url,
        "raw": response,
    }


def fetch_page_extract_with_meta(
    title: str, language_code: str, timeout_sec: float = DEFAULT_TIMEOUT_SEC
) -> dict[str, Any]:
    """
    Fetch longer plaintext extract from MediaWiki query API.
    Used when REST summary is too short for drafting.
    """
    safe_title = str(title or "").strip()
    if not safe_title:
        return {"text": "", "truncated": False}

    domain = resolve_wiki_domain(language_code)
    endpoint = f"https://{domain}/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        "redirects": "1",
        "titles": safe_title,
        "format": "json",
        "formatversion": "2",
    }
    response = _http_get_json(endpoint, params=params, timeout_sec=timeout_sec)
    pages = ((response.get("query") or {}).get("pages") or [])
    for page in pages:
        if page.get("missing"):
            continue
        extract = _clean_extract_text(str(page.get("extract") or ""))
        if extract:
            limited_text, was_truncated = _limit_source_text_with_meta(extract)
            return {"text": limited_text, "truncated": was_truncated}
    return {"text": "", "truncated": False}


def fetch_page_extract(title: str, language_code: str, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> str:
    result = fetch_page_extract_with_meta(title, language_code, timeout_sec=timeout_sec)
    return str(result.get("text") or "")


def build_script_blocks(summary_payload: dict[str, Any], max_blocks: int = DEFAULT_MAX_BLOCKS) -> list[dict[str, str]]:
    """Convert cleaned summary text into narration-friendly blocks."""
    extract = _clean_extract_text(str(summary_payload.get("extract") or ""))
    if not extract:
        return []

    sentences = _split_sentences(extract)
    if not sentences:
        return [{"text": extract}]

    limit = max(MIN_BLOCKS, min(int(max_blocks), MAX_BLOCKS))
    sentences = _expand_sentences_for_target(sentences, target_blocks=min(limit, MIN_TARGET_BLOCKS))
    packed = _pack_sentences(sentences, max_blocks=limit)
    return [{"text": item} for item in packed if item]


def fetch_wikipedia_draft(
    topic: str,
    language_code: str = "en-US",
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> dict[str, Any]:
    """
    End-to-end fetch flow for source_1.

    Returns:
      {
        "script": {"language": "...", "blocks": [...]},
        "source_meta": {...},
        "warnings": [...]
      }
    """
    warnings: list[str] = []
    language = (str(language_code or "en-US").strip() or "en-US")
    cache_key = _cache_key_for_topic(
        source="source_1",
        language_code=language,
        topic=topic,
        max_blocks=max_blocks,
    )
    cached = _load_cached_result(cache_key, ttl_sec=CACHE_TTL_SEC)
    if cached is not None:
        return cached

    search_payload = search_page(topic, language, timeout_sec=timeout_sec)
    selected = search_payload.get("selected")
    if not selected:
        raise LookupError(f"No Wikipedia result found for topic '{topic}'.")

    summary_payload = fetch_summary(selected["title"], language, timeout_sec=timeout_sec)
    if summary_payload["type"] == "disambiguation":
        warnings.append("Selected page is disambiguation-like; consider a more specific topic.")
    if len(summary_payload["extract"]) < MIN_EXTRACT_LEN:
        warnings.append("Summary is short; trying extended Wikipedia extract.")

    working_payload = dict(summary_payload)
    extended_extract_info = {"text": "", "truncated": False}
    try:
        extended_extract_info = fetch_page_extract_with_meta(
            selected["title"],
            language,
            timeout_sec=timeout_sec,
        )
    except Exception as exc:
        logger.warning("Extended extract fetch failed for '%s': %s", selected["title"], exc)
        warnings.append("Extended extract unavailable; using summary content.")
    extended_extract = str(extended_extract_info.get("text") or "").strip()
    if extended_extract and len(extended_extract) > len(summary_payload["extract"]):
        working_payload["extract"] = extended_extract
        warnings.append("Draft expanded using extended Wikipedia content.")
    elif not extended_extract and len(summary_payload["extract"]) < MIN_EXTRACT_LEN:
        warnings.append("Extended extract unavailable; using short summary only.")

    if bool(extended_extract_info.get("truncated")):
        warnings.append(
            "Source text was clipped by safety limits to avoid oversized drafts."
        )

    blocks = build_script_blocks(working_payload, max_blocks=max_blocks)
    min_target = min(max(int(max_blocks), MIN_BLOCKS), MIN_TARGET_BLOCKS)

    # Fallback expansion when sentence splitting produced fewer blocks than target.
    if len(blocks) < min_target:
        expanded = _expand_sentences_for_target(
            _split_sentences(working_payload["extract"]),
            target_blocks=min_target,
        )
        repacked = _pack_sentences(expanded, max_blocks=max(int(max_blocks), MIN_BLOCKS))
        if repacked:
            blocks = [{"text": item} for item in repacked if item]

    if not blocks:
        raise RuntimeError("Wikipedia summary was empty after cleanup.")

    fetched_at = datetime.now(timezone.utc).isoformat()
    source_meta = {
        "source": "source_1",
        "title": summary_payload["title"],
        "url": summary_payload["canonical_url"],
        "lang": language,
        "fetched_at": fetched_at,
        "confidence": _confidence_from_match(topic, selected),
    }
    source_draft, draft_stats = _build_source_draft(
        topic=topic,
        language=language,
        title=summary_payload["title"],
        canonical_url=summary_payload["canonical_url"],
        fetched_at=fetched_at,
        extract=working_payload["extract"],
        warnings=warnings,
    )

    result = {
        "script": {
            "language": language,
            "blocks": blocks,
        },
        "source_draft": source_draft,
        "draft_stats": draft_stats,
        "source_meta": source_meta,
        "warnings": warnings,
    }
    _save_cached_result(cache_key, result)
    return result


def _http_get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> dict[str, Any]:
    query = urlencode(params or {}, doseq=True)
    full_url = f"{url}?{query}" if query else url
    request = Request(
        full_url,
        headers={
            "Accept": "application/json",
            "User-Agent": WIKIPEDIA_USER_AGENT,
        },
        method="GET",
    )
    for attempt in range(HTTP_RETRY_COUNT + 1):
        try:
            with urlopen(request, timeout=timeout_sec) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                payload = response.read().decode(charset, errors="replace")
            try:
                return json.loads(payload)
            except json.JSONDecodeError as exc:
                logger.exception("Wikipedia API returned invalid JSON: %s", full_url)
                raise RuntimeError("Wikipedia API returned invalid JSON.") from exc
        except HTTPError as exc:
            retriable = exc.code in {429, 500, 502, 503, 504}
            if attempt < HTTP_RETRY_COUNT and retriable:
                _sleep_backoff(attempt)
                continue
            raise RuntimeError(f"Wikipedia API HTTP error {exc.code} for {full_url}") from exc
        except (URLError, TimeoutError) as exc:
            if attempt < HTTP_RETRY_COUNT:
                _sleep_backoff(attempt)
                continue
            reason = getattr(exc, "reason", str(exc))
            raise RuntimeError(f"Wikipedia API network error: {reason}") from exc

    raise RuntimeError("Wikipedia API request failed after retries.")


def _sleep_backoff(attempt: int):
    delay = HTTP_RETRY_BACKOFF_SEC * (2**attempt)
    time.sleep(delay)


def _cache_key_for_topic(
    *,
    source: str,
    language_code: str,
    topic: str,
    max_blocks: int,
) -> str:
    canonical = {
        "source": source,
        "language": (language_code or "").strip().lower(),
        "topic": _normalize_topic(topic).casefold(),
        "max_blocks": int(max_blocks),
        "version": 4,
    }
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_file_path(cache_key: str) -> Path:
    WIKI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return WIKI_CACHE_DIR / f"{cache_key}.json"


def _load_cached_result(cache_key: str, ttl_sec: int) -> dict[str, Any] | None:
    path = _cache_file_path(cache_key)
    if not path.exists():
        return None
    age_sec = time.time() - path.stat().st_mtime
    if age_sec > ttl_sec:
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read wiki cache file: %s", path)
        return None

    if not isinstance(data, dict) or "script" not in data or "source_meta" not in data:
        return None

    result = json.loads(json.dumps(data, ensure_ascii=False))
    source_meta = result.get("source_meta") or {}
    source_meta["cache_hit"] = True
    result["source_meta"] = source_meta
    return result


def _save_cached_result(cache_key: str, payload: dict[str, Any]):
    path = _cache_file_path(cache_key)
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.warning("Failed to write wiki cache file: %s", path)


def _build_source_draft(
    *,
    topic: str,
    language: str,
    title: str,
    canonical_url: str,
    fetched_at: str,
    extract: str,
    warnings: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    clean_extract = _clean_extract_text(extract)
    sentences = _split_sentences(clean_extract)
    if not sentences and clean_extract:
        sentences = [clean_extract]

    sentence_span = _section_sentence_span(len(sentences))
    sections: list[dict[str, Any]] = []
    for start in range(0, len(sentences), sentence_span):
        chunk = " ".join(sentences[start : start + sentence_span]).strip()
        if not chunk:
            continue
        rank = len(sections) + 1
        sections.append(
            {
                "section_id": f"s{rank:03d}",
                "title": "Overview" if rank == 1 else f"Section {rank}",
                "text": chunk,
                "rank": rank,
                "source_url": canonical_url,
                "token_estimate": _estimate_tokens(chunk),
            }
        )
        if len(sections) >= MAX_SOURCE_SECTIONS:
            break

    if not sections and clean_extract:
        sections = [
            {
                "section_id": "s001",
                "title": "Overview",
                "text": clean_extract,
                "rank": 1,
                "source_url": canonical_url,
                "token_estimate": _estimate_tokens(clean_extract),
            }
        ]

    word_count = len(clean_extract.split())
    char_count = len(clean_extract)
    sentence_count = len(sentences)
    section_count = len(sections)
    draft_id_seed = "|".join(
        [
            "source_1",
            language,
            _normalize_topic(topic).casefold(),
            title.strip().casefold(),
            canonical_url.strip(),
        ]
    )
    draft_id = hashlib.sha1(draft_id_seed.encode("utf-8")).hexdigest()[:16]

    source_draft = {
        "id": draft_id,
        "source": "source_1",
        "topic_query": _normalize_topic(topic),
        "language": language,
        "title": title,
        "source_url": canonical_url,
        "fetched_at": fetched_at,
        "sections": sections,
        "warnings": list(warnings or []),
    }
    draft_stats = {
        "section_count": section_count,
        "sentence_count": sentence_count,
        "word_count": word_count,
        "char_count": char_count,
    }
    return source_draft, draft_stats


def _section_sentence_span(sentence_count: int) -> int:
    if sentence_count <= 16:
        return 2
    if sentence_count <= 42:
        return 3
    if sentence_count <= 96:
        return 4
    return 5


def _estimate_tokens(text: str) -> int:
    # Approximate GPT-like token count for UI statistics.
    words = len(str(text or "").split())
    return max(1, int(round(words * 1.35)))


def _normalize_topic(topic: str) -> str:
    return _SPACE_RE.sub(" ", str(topic or "").strip())


def _strip_html(text: str) -> str:
    return html.unescape(_SPACE_RE.sub(" ", _HTML_TAG_RE.sub("", text or "")).strip())


def _clean_extract_text(text: str) -> str:
    clean = html.unescape(text or "")
    clean = re.sub(r"=+\s*[^=]+?\s*=+", " ", clean)
    clean = _CITATION_RE.sub("", clean)
    clean = re.sub(r"\n{2,}", "\n", clean)
    clean = _SPACE_RE.sub(" ", clean).strip()
    return clean


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\n+", " ", text or "").strip()
    raw_parts = _SENTENCE_SPLIT_RE.split(normalized)
    return [part.strip() for part in raw_parts if part and part.strip()]


def _pack_sentences(sentences: list[str], max_blocks: int) -> list[str]:
    """
    Preserve as many short blocks as possible up to max_blocks.
    """
    if not sentences:
        return []

    cleaned = [s.strip() for s in sentences if s and s.strip()]
    if len(cleaned) <= max_blocks:
        return cleaned

    head = cleaned[:max_blocks]
    overflow = cleaned[max_blocks:]
    for extra in overflow:
        head[-1] = f"{head[-1]} {extra}".strip()
    return head


def _expand_sentences_for_target(sentences: list[str], target_blocks: int) -> list[str]:
    """
    If sentence count is low, split long sentences by soft clause delimiters.
    """
    if len(sentences) >= target_blocks:
        return sentences

    expanded: list[str] = []
    for sentence in sentences:
        expanded.extend(_split_sentence_clauses(sentence))

    if len(expanded) > len(sentences):
        return expanded
    return sentences


def _split_sentence_clauses(sentence: str) -> list[str]:
    text = (sentence or "").strip()
    if not text:
        return []
    if len(text) < 120:
        return [text]

    parts = [part.strip() for part in re.split(r",\s+|;\s+|:\s+", text) if part and part.strip()]
    if len(parts) <= 1:
        return [text]

    clauses: list[str] = []
    for part in parts:
        if not re.search(r"[.!?]$", part):
            part = f"{part}."
        clauses.append(part)
    return clauses


def _limit_source_text(text: str) -> str:
    clipped, _ = _limit_source_text_with_meta(text)
    return clipped


def _limit_source_text_with_meta(text: str) -> tuple[str, bool]:
    """
    Keep only the leading informative chunk to avoid overly long drafts.
    """
    sentences = _split_sentences(text)
    was_truncated = False
    if sentences:
        if len(sentences) > MAX_SOURCE_SENTENCES:
            was_truncated = True
        clipped = " ".join(sentences[:MAX_SOURCE_SENTENCES]).strip()
    else:
        clipped = (text or "").strip()
    if len(clipped) > MAX_SOURCE_CHARS:
        was_truncated = True
        clipped = clipped[:MAX_SOURCE_CHARS].rsplit(" ", 1)[0].strip()
    return clipped, was_truncated


def _candidate_score(topic: str, title: str, snippet: str) -> float:
    topic_norm = topic.casefold()
    title_norm = title.casefold()
    snippet_norm = snippet.casefold()
    score = 0.0

    if topic_norm == title_norm:
        score += 1.0
    if topic_norm in title_norm:
        score += 0.6
    if topic_norm in snippet_norm:
        score += 0.2
    if _is_likely_disambiguation(title, snippet):
        score -= 0.45
    return score


def _is_likely_disambiguation(title: str, snippet: str) -> bool:
    joined = f"{title} {snippet}".casefold()
    patterns = [
        "disambiguation",
        "may refer to",
        "can refer to",
        "dinh huong",
        "co the de cap",
    ]
    return any(pattern in joined for pattern in patterns)


def _confidence_from_match(topic: str, selected: dict[str, Any]) -> float:
    topic_norm = _normalize_topic(topic).casefold()
    title_norm = str(selected.get("title") or "").casefold()
    if topic_norm and topic_norm == title_norm:
        return 0.95
    if topic_norm and topic_norm in title_norm:
        return 0.85
    score = float(selected.get("score") or 0.0)
    return max(0.55, min(0.82, 0.62 + score * 0.1))
