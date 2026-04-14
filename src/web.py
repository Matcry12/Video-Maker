"""
Web UI for the Video Maker pipeline.
Run: python -m src.web
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from flask import Flask, render_template, request, jsonify, send_from_directory

from .content_bank import ContentBankStore, ingest_topics_from_wikipedia
from .content_bank.selector import (
    SUPPORTED_SELECTION_MODES,
    build_script_from_facts,
    select_facts,
)
from .content_sources import (
    extract_source_units_from_draft,
    fetch_wikipedia_draft,
    rank_interest_candidates,
    refine_draft_with_groq,
    to_rank_candidates,
)
from .content_sources.fact_script_writer import write_script_from_facts
from .content_sources.script_lint import lint_script
from .content_sources import ranked_items_to_facts
from .content_sources.wikipedia_source import _split_sentences
from .manager import VideoManager, PROJECT_ROOT, OUTPUT_DIR, Profile

logger = logging.getLogger(__name__)


def _load_local_env_file(path: Path):
    if not path.exists():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
    except Exception:
        logger.warning("Failed to load local .env file: %s", path)


_load_local_env_file(PROJECT_ROOT / ".env")

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)

# Global state
jobs = {}  # job_id -> {status, progress, output, error}
CONTENT_BANK_STORE = ContentBankStore()

VOICES = [
    {"id": "NamMinh", "label": "NamMinh (Vietnamese, male)"},
    {"id": "HoaiMy", "label": "HoaiMy (Vietnamese, female)"},
    {"id": "Guy", "label": "Guy (English, male)"},
    {"id": "Aria", "label": "Aria (English, female)"},
]



def create_manager():
    logger.info("Initializing VideoManager...")
    mgr = VideoManager()
    logger.info("VideoManager ready.")
    return mgr


def api_error(
    message: str,
    *,
    code: str = "UNKNOWN_ERROR",
    status: int = 400,
    details: str | None = None,
    hint: str | None = None,
):
    payload = {
        "error": {
            "code": code,
            "message": message,
        }
    }
    if details:
        payload["error"]["details"] = details
    if hint:
        payload["error"]["hint"] = hint
    return jsonify(payload), status


def output_to_payload(path: Path) -> dict:
    stat = path.stat()
    created_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    encoded_name = quote(path.name)
    audio_path = path.with_suffix(".mp3")
    return {
        "filename": path.name,
        "size_bytes": stat.st_size,
        "created_at": created_dt.isoformat(),
        "created_ts": stat.st_mtime,
        "url": f"/output/{encoded_name}",
        "download_url": f"/output/{encoded_name}?download=1",
        "audio_filename": audio_path.name if audio_path.exists() else None,
        "audio_url": f"/output/{quote(audio_path.name)}" if audio_path.exists() else None,
        "audio_download_url": f"/output/{quote(audio_path.name)}?download=1" if audio_path.exists() else None,
    }


def _parse_bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _model_to_dict(model_like):
    if hasattr(model_like, "model_dump"):
        return model_like.model_dump()
    if hasattr(model_like, "dict"):
        return model_like.dict()
    return dict(model_like)


DEFAULT_SOURCE_FETCH_BLOCKS = 8
MAX_SOURCE_FETCH_BLOCKS = 24
DEFAULT_TOP_SOURCE_UNITS = 24
DRAFTS_DIR = PROJECT_ROOT / "content_drafts"


def _normalize_source_draft_input(
    source_draft_raw: Any,
    *,
    fallback_language: str = "vi-VN",
) -> dict[str, Any]:
    if not isinstance(source_draft_raw, dict):
        raise ValueError("source_draft must be an object.")

    sections_raw = source_draft_raw.get("sections")
    if not isinstance(sections_raw, list):
        raise ValueError("source_draft.sections must be an array.")

    normalized_sections: list[dict[str, Any]] = []
    for idx, section in enumerate(sections_raw):
        if not isinstance(section, dict):
            continue
        text = str(section.get("text") or "").strip()
        if not text:
            continue
        rank_default = len(normalized_sections) + 1
        rank_raw = section.get("rank", rank_default)
        try:
            rank = int(rank_raw)
        except (TypeError, ValueError):
            rank = rank_default
        normalized_sections.append(
            {
                "section_id": str(section.get("section_id") or f"s{rank_default:03d}").strip()
                or f"s{rank_default:03d}",
                "title": str(section.get("title") or f"Section {rank_default}").strip()
                or f"Section {rank_default}",
                "text": text,
                "rank": max(1, rank),
                "source_url": str(section.get("source_url") or source_draft_raw.get("source_url") or "").strip(),
                "token_estimate": section.get("token_estimate"),
            }
        )

    if not normalized_sections:
        raise ValueError("source_draft.sections is empty.")

    language = str(source_draft_raw.get("language") or fallback_language).strip() or fallback_language
    normalized = {
        "id": str(source_draft_raw.get("id") or "").strip(),
        "source": str(source_draft_raw.get("source") or "source_1").strip() or "source_1",
        "topic_query": str(source_draft_raw.get("topic_query") or "").strip(),
        "language": language,
        "title": str(source_draft_raw.get("title") or "").strip(),
        "source_url": str(source_draft_raw.get("source_url") or "").strip(),
        "fetched_at": str(source_draft_raw.get("fetched_at") or "").strip(),
        "sections": normalized_sections,
        "warnings": [str(item) for item in (source_draft_raw.get("warnings") or []) if str(item).strip()],
    }
    return normalized


def _source_draft_to_raw_script(
    source_draft: dict[str, Any],
    *,
    language: str | None = None,
) -> dict[str, Any]:
    script_language = str(language or source_draft.get("language") or "vi-VN").strip() or "vi-VN"
    topic = str(source_draft.get("topic_query") or "").strip()
    source_url = str(source_draft.get("source_url") or "").strip()
    blocks: list[dict[str, Any]] = []

    for section in source_draft.get("sections") or []:
        if not isinstance(section, dict):
            continue
        text = str(section.get("text") or "").strip()
        if not text:
            continue
        block = {"text": text}
        section_id = str(section.get("section_id") or "").strip()
        section_title = str(section.get("title") or "").strip()
        section_source_url = str(section.get("source_url") or source_url).strip()
        if section_id:
            block["source_section_id"] = section_id
        if section_title:
            block["source_section_title"] = section_title
        if topic:
            block["source_topic"] = topic
        if section_source_url:
            block["source_url"] = section_source_url
        blocks.append(block)

    if not blocks:
        raise ValueError("source_draft does not contain usable section text.")

    return {
        "language": script_language,
        "blocks": blocks,
    }


def _derive_target_blocks(section_count: int) -> int:
    if section_count <= 3:
        return 3
    if section_count <= 8:
        return section_count
    return max(4, min(12, round(section_count * 0.55)))


def _parse_section_selector(selector_raw: Any) -> list[int]:
    selector = str(selector_raw or "").strip()
    if not selector:
        return []

    selected: set[int] = set()
    for chunk in selector.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            try:
                start = int(start_raw.strip())
                end = int(end_raw.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid section range '{token}'.") from exc
            if start < 1 or end < 1:
                raise ValueError(f"Invalid section range '{token}'.")
            low, high = sorted((start, end))
            selected.update(range(low, high + 1))
            continue
        try:
            index = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid section number '{token}'.") from exc
        if index < 1:
            raise ValueError(f"Invalid section number '{token}'.")
        selected.add(index)
    return sorted(selected)


def _filter_source_draft_sections(source_draft: dict[str, Any], selected_ranks: list[int]) -> dict[str, Any]:
    if not selected_ranks:
        return source_draft

    selected_set = set(selected_ranks)
    filtered_sections: list[dict[str, Any]] = []
    for idx, section in enumerate(source_draft.get("sections") or []):
        if not isinstance(section, dict):
            continue
        rank = section.get("rank")
        try:
            normalized_rank = int(rank)
        except (TypeError, ValueError):
            normalized_rank = idx + 1
        if normalized_rank in selected_set:
            filtered_sections.append(section)

    if not filtered_sections:
        raise ValueError("Section selector did not match any fetched sections.")

    filtered = dict(source_draft)
    filtered["sections"] = filtered_sections
    return filtered


def _sanitize_filename_token(value: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(value or "").strip())
    safe = safe.strip("_")
    return safe or "draft"


@app.route("/")
def index():
    # List existing scripts
    scripts_dir = PROJECT_ROOT / "json_scripts"
    scripts = sorted(scripts_dir.glob("*.json")) if scripts_dir.exists() else []
    script_names = [s.stem for s in scripts]

    # List existing outputs
    outputs = sorted(OUTPUT_DIR.glob("*.mp4")) if OUTPUT_DIR.exists() else []
    output_names = [o.name for o in outputs]

    subtitle_presets = ["minimal", "energetic", "cinematic"]
    default_subtitle_preset = "minimal"
    default_subtitle_alignment_mode = "corrected"
    profile_path = PROJECT_ROOT / "profiles" / "default.json"
    if profile_path.exists():
        try:
            profile_raw = json.loads(profile_path.read_text())
            profile = Profile(**profile_raw)
            subtitle_presets = sorted(profile.subtitle.presets.keys())
            default_subtitle_preset = profile.subtitle.default_preset
            default_subtitle_alignment_mode = profile.subtitle.default_alignment_mode
        except Exception:
            logger.exception("Failed to load subtitle presets from profile")

    return render_template(
        "index.html",
        voices=VOICES,
        scripts=script_names,
        outputs=output_names,
        subtitle_presets=subtitle_presets,
        default_subtitle_preset=default_subtitle_preset,
        default_subtitle_alignment_mode=default_subtitle_alignment_mode,
    )


@app.route("/api/scripts/<name>")
def get_script(name):
    path = PROJECT_ROOT / "json_scripts" / f"{name}.json"
    if not path.exists():
        return api_error(
            "Script not found.",
            code="SCRIPT_NOT_FOUND",
            status=404,
            hint="Select an existing script from the list or save a new script name first.",
        )
    try:
        return jsonify(json.loads(path.read_text()))
    except Exception as exc:
        logger.exception("Failed to parse script: %s", path)
        return api_error(
            "Script file is invalid JSON.",
            code="INVALID_SCRIPT_JSON",
            status=400,
            details=str(exc),
            hint="Fix JSON content in the script file, then reload.",
        )


@app.route("/api/scripts")
def list_scripts():
    scripts_dir = PROJECT_ROOT / "json_scripts"
    scripts = sorted(scripts_dir.glob("*.json")) if scripts_dir.exists() else []
    return jsonify([s.stem for s in scripts])


@app.route("/api/scripts/<name>", methods=["POST"])
def save_script(name):
    data = request.get_json()
    if data is None:
        return api_error(
            "Request body must be JSON.",
            code="INVALID_REQUEST_JSON",
            status=400,
            hint="Send a valid JSON script payload.",
        )
    path = PROJECT_ROOT / "json_scripts" / f"{name}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return jsonify({"ok": True})
    except Exception as exc:
        logger.exception("Failed to save script: %s", path)
        return api_error(
            "Failed to save script.",
            code="SAVE_SCRIPT_FAILED",
            status=500,
            details=str(exc),
            hint="Check file permissions and free disk space, then try again.",
        )


@app.route("/api/scripts/<name>", methods=["DELETE"])
def delete_script(name):
    path = PROJECT_ROOT / "json_scripts" / f"{name}.json"
    try:
        if path.exists():
            path.unlink()
        return jsonify({"ok": True})
    except Exception as exc:
        logger.exception("Failed to delete script: %s", path)
        return api_error(
            "Failed to delete script.",
            code="DELETE_SCRIPT_FAILED",
            status=500,
            details=str(exc),
            hint="Close any process using this file and retry.",
        )


@app.route("/api/content/bank/ingest", methods=["POST"])
def bank_ingest_topics():
    data = request.get_json()
    if not isinstance(data, dict):
        return api_error(
            "Request body must be JSON.",
            code="INVALID_REQUEST_JSON",
            status=400,
            hint="Send topics and optional language/max_topics in JSON.",
        )

    topics_raw = data.get("topics")
    if isinstance(topics_raw, str):
        topics = [line.strip() for line in topics_raw.splitlines() if line.strip()]
    elif isinstance(topics_raw, list):
        topics = [str(item).strip() for item in topics_raw if str(item).strip()]
    else:
        topics = []

    if not topics:
        return api_error(
            "topics is required.",
            code="TOPICS_REQUIRED",
            status=400,
            hint="Provide topics as an array or newline-separated string.",
        )

    language = str(data.get("language") or "en-US").strip() or "en-US"
    max_topics_raw = data.get("max_topics", 20)
    try:
        max_topics = int(max_topics_raw)
    except (TypeError, ValueError):
        return api_error(
            "max_topics must be an integer.",
            code="INVALID_MAX_TOPICS",
            status=400,
            hint="Use a value between 1 and 50.",
        )
    if max_topics < 1 or max_topics > 50:
        return api_error(
            "max_topics out of range.",
            code="INVALID_MAX_TOPICS",
            status=400,
            hint="Use a value between 1 and 50.",
        )

    extract_facts = _parse_bool(data.get("extract_facts", True))
    facts_per_topic_raw = data.get("facts_per_topic_target", 8)
    try:
        facts_per_topic_target = int(facts_per_topic_raw)
    except (TypeError, ValueError):
        return api_error(
            "facts_per_topic_target must be an integer.",
            code="INVALID_FACTS_PER_TOPIC",
            status=400,
            hint="Use a value between 1 and 20.",
        )
    if facts_per_topic_target < 1 or facts_per_topic_target > 20:
        return api_error(
            "facts_per_topic_target out of range.",
            code="INVALID_FACTS_PER_TOPIC",
            status=400,
            hint="Use a value between 1 and 20.",
        )

    fact_prompt_override = str(data.get("fact_prompt_override") or "").strip() or None
    if fact_prompt_override and len(fact_prompt_override) > 6000:
        return api_error(
            "fact_prompt_override is too long.",
            code="INVALID_PROMPT_OVERRIDE",
            status=400,
            hint="Keep prompt override under 6000 characters.",
        )

    near_dup_raw = data.get("near_dup_threshold", 0.88)
    try:
        near_dup_threshold = float(near_dup_raw)
    except (TypeError, ValueError):
        return api_error(
            "near_dup_threshold must be a number.",
            code="INVALID_NEAR_DUP_THRESHOLD",
            status=400,
            hint="Use a decimal value between 0.6 and 0.99.",
        )
    if near_dup_threshold < 0.6 or near_dup_threshold > 0.99:
        return api_error(
            "near_dup_threshold out of range.",
            code="INVALID_NEAR_DUP_THRESHOLD",
            status=400,
            hint="Use a decimal value between 0.6 and 0.99.",
        )

    try:
        run = ingest_topics_from_wikipedia(
            CONTENT_BANK_STORE,
            topics,
            language=language,
            max_topics=max_topics,
            source="source_1",
            extract_facts=extract_facts,
            facts_per_topic_target=facts_per_topic_target,
            fact_prompt_override=fact_prompt_override,
            near_dup_threshold=near_dup_threshold,
        )
    except Exception as exc:
        logger.exception("Topic ingest failed.")
        return api_error(
            "Failed to ingest topics into content bank.",
            code="BANK_INGEST_FAILED",
            status=502,
            details=str(exc),
            hint="Check network/source availability and retry.",
        )

    return jsonify(
        {
            "ok": True,
            "accepted_topics": run.accepted_topics,
            "skipped_topics": run.skipped_topics,
            "created_topics": run.created_topics,
            "updated_topics": run.updated_topics,
            "extracted_facts": run.extracted_facts,
            "saved_facts": run.saved_facts,
            "created_facts": run.created_facts,
            "updated_facts": run.updated_facts,
            "deduped_facts": run.deduped_facts,
            "items": [_model_to_dict(item) for item in run.items],
            "extraction_meta": run.extraction_meta,
            "warnings": run.warnings,
        }
    )


@app.route("/api/content/bank/facts", methods=["GET"])
def bank_list_facts():
    status = str(request.args.get("status") or "").strip().lower() or None
    topic_id = str(request.args.get("topic_id") or "").strip() or None
    limit_raw = request.args.get("limit", "100")
    offset_raw = request.args.get("offset", "0")
    try:
        limit = int(limit_raw)
        offset = int(offset_raw)
    except (TypeError, ValueError):
        return api_error(
            "limit and offset must be integers.",
            code="INVALID_PAGINATION",
            status=400,
            hint="Use numeric query params, e.g. limit=100&offset=0.",
        )
    if limit < 1 or limit > 500 or offset < 0:
        return api_error(
            "Pagination out of range.",
            code="INVALID_PAGINATION",
            status=400,
            hint="Use limit 1-500 and offset >= 0.",
        )

    facts, total = CONTENT_BANK_STORE.list_facts(
        status=status,
        topic_id=topic_id,
        limit=limit,
        offset=offset,
    )
    return jsonify(
        {
            "ok": True,
            "facts": [_model_to_dict(item) for item in facts],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@app.route("/api/content/bank/compose", methods=["POST"])
def bank_compose_script():
    data = request.get_json()
    if not isinstance(data, dict):
        return api_error(
            "Request body must be JSON.",
            code="INVALID_REQUEST_JSON",
            status=400,
            hint="Send compose settings as JSON.",
        )

    selection_mode = str(data.get("selection_mode") or "balanced").strip().lower()
    if selection_mode not in SUPPORTED_SELECTION_MODES:
        return api_error(
            "Unsupported selection_mode.",
            code="INVALID_SELECTION_MODE",
            status=400,
            hint="Use one of: top, balanced, random_weighted.",
        )

    pick_topics_raw = data.get("pick_topics_count", 3)
    pick_facts_raw = data.get("pick_facts_count", data.get("target_blocks", pick_topics_raw))
    try:
        pick_topics_count = int(pick_topics_raw)
        pick_facts_count = int(pick_facts_raw)
    except (TypeError, ValueError):
        return api_error(
            "pick_topics_count and pick_facts_count must be integers.",
            code="INVALID_PICK_COUNTS",
            status=400,
            hint="Use numeric values between 1 and 100.",
        )
    if pick_topics_count < 1 or pick_topics_count > 50 or pick_facts_count < 1 or pick_facts_count > 100:
        return api_error(
            "pick counts out of range.",
            code="INVALID_PICK_COUNTS",
            status=400,
            hint="Use pick_topics_count 1-50 and pick_facts_count 1-100.",
        )

    language = str(data.get("language") or "en-US").strip() or "en-US"
    exclude_used = _parse_bool(data.get("exclude_used", True))
    min_score_raw = data.get("min_score", None)
    min_score = None
    if min_score_raw is not None and str(min_score_raw).strip() != "":
        try:
            min_score = float(min_score_raw)
        except (TypeError, ValueError):
            return api_error(
                "min_score must be a number.",
                code="INVALID_MIN_SCORE",
                status=400,
                hint="Use decimal value 0.0 to 1.0.",
            )
        if min_score < 0.0 or min_score > 1.0:
            return api_error(
                "min_score out of range.",
                code="INVALID_MIN_SCORE",
                status=400,
                hint="Use decimal value 0.0 to 1.0.",
            )

    topic_ids_raw = data.get("topic_ids")
    if isinstance(topic_ids_raw, str):
        topic_ids = {part.strip() for part in topic_ids_raw.split(",") if part.strip()}
    elif isinstance(topic_ids_raw, list):
        topic_ids = {str(item).strip() for item in topic_ids_raw if str(item).strip()}
    else:
        topic_ids = set()

    query = str(data.get("query") or "").strip().casefold()
    facts = CONTENT_BANK_STORE.load_facts()

    filtered_facts = []
    for fact in facts:
        if topic_ids and fact.topic_id not in topic_ids:
            continue
        if min_score is not None and float(fact.score or 0.0) < min_score:
            continue
        if query:
            haystack = " ".join(
                [
                    str(fact.fact_text or ""),
                    str(fact.hook_text or ""),
                    str(fact.topic_label or ""),
                ]
            ).casefold()
            if query not in haystack:
                continue
        filtered_facts.append(fact)

    selected, selection_meta = select_facts(
        filtered_facts,
        pick_topics_count=pick_topics_count,
        pick_facts_count=pick_facts_count,
        selection_mode=selection_mode,
        exclude_used=exclude_used,
    )
    script = build_script_from_facts(selected, language=language)
    fact_ids = [fact.fact_id for fact in selected]
    warnings = []
    if not selected:
        warnings.append("No facts matched current filters.")

    return jsonify(
        {
            "ok": True,
            "script": script,
            "selected_facts": [_model_to_dict(item) for item in selected],
            "selected_fact_ids": fact_ids,
            "selection_meta": selection_meta,
            "available_after_filter": len(filtered_facts),
            "warnings": warnings,
        }
    )


@app.route("/api/content/bank/facts/mark-used", methods=["POST"])
def bank_mark_facts_used():
    data = request.get_json()
    if not isinstance(data, dict):
        return api_error(
            "Request body must be JSON.",
            code="INVALID_REQUEST_JSON",
            status=400,
            hint="Send fact_ids and optional video_id as JSON.",
        )

    fact_ids_raw = data.get("fact_ids")
    if isinstance(fact_ids_raw, str):
        fact_ids = [item.strip() for item in fact_ids_raw.split(",") if item.strip()]
    elif isinstance(fact_ids_raw, list):
        fact_ids = [str(item).strip() for item in fact_ids_raw if str(item).strip()]
    else:
        fact_ids = []

    if not fact_ids:
        return api_error(
            "fact_ids is required.",
            code="FACT_IDS_REQUIRED",
            status=400,
            hint="Provide fact_ids as an array or comma-separated string.",
        )

    video_id = str(data.get("video_id") or "").strip() or None
    updated_count = CONTENT_BANK_STORE.mark_facts_used(fact_ids, video_id=video_id)
    return jsonify(
        {
            "ok": True,
            "updated_count": updated_count,
        }
    )


@app.route("/api/content/fetch", methods=["POST"])
def fetch_content():
    data = request.get_json()
    if not isinstance(data, dict):
        return api_error(
            "Request body must be JSON.",
            code="INVALID_REQUEST_JSON",
            status=400,
            hint="Send source, topic, and optional prompt/debug fields as JSON.",
        )

    source = str(data.get("source") or "source_1").strip().lower()
    if source not in {"source_1", "wikipedia"}:
        return api_error(
            "Unsupported content source.",
            code="UNSUPPORTED_SOURCE",
            status=400,
            hint="Use source_1 (Wikipedia).",
        )

    topic = str(data.get("topic") or "").strip()
    if not topic:
        return api_error(
            "Topic is required.",
            code="TOPIC_REQUIRED",
            status=400,
            hint="Enter a topic keyword, then fetch again.",
        )

    language = str(data.get("language") or "vi-VN").strip() or "vi-VN"
    section_selector_raw = data.get("section_selector")
    try:
        selected_section_ranks = _parse_section_selector(section_selector_raw)
    except ValueError as exc:
        return api_error(
            "section_selector is invalid.",
            code="INVALID_SECTION_SELECTOR",
            status=400,
            details=str(exc),
            hint="Use values like 1-4, 5, 10.",
        )

    llm_provider = str(data.get("llm_provider") or "").strip().lower()
    if llm_provider not in {"", "groq", "ollama"}:
        return api_error(
            "Unsupported llm_provider.",
            code="UNSUPPORTED_LLM_PROVIDER",
            status=400,
            hint="Set llm_provider to 'groq' or 'ollama'.",
        )
    if not llm_provider:
        llm_provider = "groq"

    max_blocks_raw = data.get("max_blocks", data.get("target_blocks", DEFAULT_SOURCE_FETCH_BLOCKS))
    try:
        source_max_blocks = int(max_blocks_raw)
    except (TypeError, ValueError):
        return api_error(
            "max_blocks must be an integer.",
            code="INVALID_MAX_BLOCKS",
            status=400,
            hint="Use a number between 1 and 24.",
        )
    if source_max_blocks < 1 or source_max_blocks > MAX_SOURCE_FETCH_BLOCKS:
        return api_error(
            "max_blocks out of range.",
            code="INVALID_MAX_BLOCKS",
            status=400,
            hint=f"Use a value between 1 and {MAX_SOURCE_FETCH_BLOCKS}.",
        )

    rank_top_k_raw = data.get("rank_top_k", DEFAULT_TOP_SOURCE_UNITS)
    try:
        rank_top_k = int(rank_top_k_raw)
    except (TypeError, ValueError):
        return api_error(
            "rank_top_k must be an integer.",
            code="INVALID_TOP_K",
            status=400,
            hint="Use a number between 1 and 60.",
        )
    if rank_top_k < 1 or rank_top_k > 60:
        return api_error(
            "rank_top_k out of range.",
            code="INVALID_TOP_K",
            status=400,
            hint="Use a value between 1 and 60.",
        )

    run_interest_rank = _parse_bool(data.get("run_interest_rank", True))

    try:
        raw_result = fetch_wikipedia_draft(
            topic=topic,
            language_code=language,
            max_blocks=source_max_blocks,
        )
    except LookupError as exc:
        return api_error(
            "No matching Wikipedia content found.",
            code="SOURCE_NOT_FOUND",
            status=404,
            details=str(exc),
            hint="Try a more specific keyword or change language.",
        )
    except Exception as exc:
        logger.exception("Content fetch failed for topic '%s'", topic)
        return api_error(
            "Failed to fetch content from source_1.",
            code="SOURCE_FETCH_FAILED",
            status=502,
            details=str(exc),
            hint="Check internet connection and retry.",
        )

    warnings = list(raw_result.get("warnings") or [])
    warnings.append("Fetch completed. Use Draft Handling to save raw JSON or transform via LLM.")
    source_draft = raw_result.get("source_draft") or {}
    if selected_section_ranks:
        try:
            source_draft = _filter_source_draft_sections(source_draft, selected_section_ranks)
        except ValueError as exc:
            return api_error(
                "No matching sections found for selection.",
                code="SECTION_SELECTION_EMPTY",
                status=400,
                details=str(exc),
                hint="Check the section numbers shown in Draft Sections and retry.",
            )
        warnings.append(
            f"Filtered to sections: {', '.join(str(item) for item in selected_section_ranks)}."
        )

    raw_script = _source_draft_to_raw_script(source_draft, language=language)
    filtered_sections = source_draft.get("sections") or []
    draft_stats = {
        "section_count": len(filtered_sections),
        "sentence_count": sum(
            len(_split_sentences(str(section.get("text") or "").strip()))
            for section in filtered_sections
            if isinstance(section, dict)
        ),
        "word_count": sum(
            len(str(section.get("text") or "").split())
            for section in filtered_sections
            if isinstance(section, dict)
        ),
        "char_count": sum(
            len(str(section.get("text") or ""))
            for section in filtered_sections
            if isinstance(section, dict)
        ),
        "selected_section_ranks": selected_section_ranks,
    }
    include_raw = _parse_bool(data.get("include_raw", True))

    extraction_result = extract_source_units_from_draft(
        source_draft,
        keep_top_k=rank_top_k,
    )
    ranked_candidates_payload = []
    ranking_meta: dict[str, Any] = {
        "provider": llm_provider,
        "status": "skipped",
        "model": os.getenv("GROQ_MODEL", "").strip() or None,
        "candidate_count": len(extraction_result.get("top_units") or []),
        "batch_count": 0,
        "parsed_batches": 0,
        "repaired_batches": 0,
        "failed_batches": 0,
    }
    ranking_warnings: list[str] = []

    if llm_provider == "ollama":
        interest_model = (
            os.getenv("OLLAMA_INTEREST_MODEL", "").strip()
            or os.getenv("OLLAMA_MODEL", "").strip()
            or "gemma4:e2b"
        )
    else:
        interest_model = (
            os.getenv("GROQ_INTEREST_MODEL", "").strip()
            or os.getenv("GROQ_MODEL", "").strip()
            or "llama-3.1-8b-instant"
        )
    ranking_meta["model"] = interest_model

    if run_interest_rank and extraction_result.get("top_units"):
        try:
            ranking_result = rank_interest_candidates(
                to_rank_candidates(extraction_result["top_units"]),
                model=interest_model,
                language=language,
                provider=llm_provider,
            )
            ranked_candidates_payload = ranking_result.get("items") or []
            ranking_meta.update(ranking_result.get("meta") or {})
            ranking_meta["provider"] = llm_provider
            ranking_meta["status"] = (
                "ok"
                if int(ranking_meta.get("failed_batches") or 0) == 0
                else "partial_fallback"
            )
            ranking_warnings.extend(ranking_result.get("warnings") or [])
        except Exception as exc:
            logger.warning("Interest ranking failed for topic '%s': %s", topic, exc)
            ranking_meta["status"] = "fallback_local_only"
            ranking_meta["error"] = str(exc)
            ranking_warnings.append("Interest ranking failed. Using local extraction scores only.")
    else:
        ranking_meta["status"] = "skipped_by_request" if not run_interest_rank else "no_candidates"

    warnings.extend(ranking_warnings)

    response_payload = {
        "ok": True,
        "script": raw_script,
        "source_draft": source_draft,
        "draft_stats": draft_stats,
        "source_meta": raw_result["source_meta"],
        "llm_meta": {
            "provider": llm_provider,
            "status": "not_run_fetch_only",
            "prompt_mode": "n/a",
        },
        "source_documents": extraction_result.get("documents") or [],
        "source_units": extraction_result.get("units") or [],
        "top_source_units": extraction_result.get("top_units") or [],
        "source_unit_meta": extraction_result.get("meta") or {},
        "ranked_candidates": ranked_candidates_payload,
        "ranking_meta": ranking_meta,
        "warnings": warnings,
        "draft_mode_effective": "raw",
        "source_max_blocks": source_max_blocks,
        "run_interest_rank": run_interest_rank,
        "rank_top_k": rank_top_k,
        "section_selector": str(section_selector_raw or "").strip(),
    }
    if include_raw:
        response_payload["raw_script"] = raw_script
    return jsonify(response_payload)


@app.route("/api/content/handle", methods=["POST"])
def handle_content_draft():
    data = request.get_json()
    if not isinstance(data, dict):
        return api_error(
            "Request body must be JSON.",
            code="INVALID_REQUEST_JSON",
            status=400,
            hint="Send source_draft and handling options as JSON.",
        )

    handling_mode = str(data.get("handling_mode") or "llm_script").strip().lower()
    if handling_mode not in {"raw_json", "raw_script", "llm_script", "ranked_fact_script"}:
        return api_error(
            "handling_mode is invalid.",
            code="INVALID_HANDLING_MODE",
            status=400,
            hint="Use raw_json, raw_script, llm_script, or ranked_fact_script.",
        )

    source_draft_raw = data.get("source_draft")
    if source_draft_raw is None:
        raw_draft_json = data.get("raw_draft_json")
        if isinstance(raw_draft_json, dict):
            source_draft_raw = raw_draft_json
        elif isinstance(raw_draft_json, str) and raw_draft_json.strip():
            try:
                source_draft_raw = json.loads(raw_draft_json)
            except json.JSONDecodeError:
                return api_error(
                    "raw_draft_json is not valid JSON.",
                    code="INVALID_SOURCE_DRAFT_JSON",
                    status=400,
                    hint="Fix raw draft JSON syntax and retry.",
                )

    language = str(data.get("language") or "vi-VN").strip() or "vi-VN"
    try:
        source_draft = _normalize_source_draft_input(source_draft_raw, fallback_language=language)
    except ValueError as exc:
        return api_error(
            "source_draft is invalid.",
            code="INVALID_SOURCE_DRAFT",
            status=400,
            details=str(exc),
            hint="Fetch a source draft first or paste valid raw draft JSON.",
        )

    llm_provider = str(data.get("llm_provider") or "groq").strip().lower() or "groq"
    if llm_provider not in {"groq"}:
        return api_error(
            "Unsupported llm_provider.",
            code="UNSUPPORTED_LLM_PROVIDER",
            status=400,
            hint="Set llm_provider to 'groq'.",
        )

    prompt_override = str(data.get("prompt_override") or "").strip() or None
    if prompt_override and len(prompt_override) > 4000:
        return api_error(
            "prompt_override is too long.",
            code="INVALID_PROMPT_OVERRIDE",
            status=400,
            hint="Keep prompt override under 4000 characters.",
        )

    target_blocks_raw = data.get("target_blocks", _derive_target_blocks(len(source_draft["sections"])))
    try:
        target_blocks = int(target_blocks_raw)
    except (TypeError, ValueError):
        return api_error(
            "target_blocks must be an integer.",
            code="INVALID_MAX_BLOCKS",
            status=400,
            hint="Use a number between 1 and 12.",
        )
    if target_blocks < 1 or target_blocks > 12:
        return api_error(
            "target_blocks out of range.",
            code="INVALID_MAX_BLOCKS",
            status=400,
            hint="Use a value between 1 and 12.",
        )

    warnings = list(source_draft.get("warnings") or [])
    raw_script = _source_draft_to_raw_script(source_draft, language=language)
    script = None
    llm_meta: dict[str, Any] | None = None

    if handling_mode == "raw_json":
        llm_meta = {
            "provider": llm_provider,
            "status": "skipped_raw_json",
            "prompt_mode": "raw_json",
        }
    elif handling_mode == "raw_script":
        script = raw_script
        llm_meta = {
            "provider": llm_provider,
            "status": "skipped_raw_script",
            "prompt_mode": "raw_script",
            "target_blocks": target_blocks,
        }
    elif handling_mode == "llm_script":
        try:
            llm_result = refine_draft_with_groq(
                raw_script=raw_script,
                language=language,
                style=str(data.get("style") or "").strip() or None,
                prompt_override=prompt_override,
                target_blocks=target_blocks,
                length_target="medium",
            )
            script = llm_result["script"]
            llm_meta = llm_result["meta"]
            llm_meta["status"] = "ok"
        except Exception as exc:
            logger.warning("Groq refinement failed; fallback to raw script: %s", exc)
            script = raw_script
            warnings.append(
                "LLM refinement unavailable. Returned raw draft so you can edit manually."
            )
            reason_text = str(exc).strip()
            reason_lower = reason_text.lower()
            if "groq_api_key" in reason_lower:
                warnings.append("Groq API key is missing in runtime environment.")
            elif "http error 401" in reason_lower or "http error 403" in reason_lower:
                warnings.append("Groq API key was rejected (401/403). Check your key and model access.")
            elif "timed out" in reason_lower:
                warnings.append("Groq request timed out. Retry in a moment.")
            llm_meta = {
                "provider": llm_provider,
                "status": "fallback_raw",
                "error": str(exc),
                "prompt_mode": "override" if prompt_override else "default",
                "target_blocks": target_blocks,
                "length_target": "medium",
            }
    elif handling_mode == "ranked_fact_script":
        ranked_items = data.get("ranked_candidates")
        rank_candidates_list = []
        if not ranked_items:
            try:
                extraction_result = extract_source_units_from_draft(source_draft, keep_top_k=24)
                top_units = extraction_result.get("top_units") or []
                if top_units:
                    interest_model = os.getenv("GROQ_INTEREST_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
                    rank_candidates_list = to_rank_candidates(top_units)
                    ranking_result = rank_interest_candidates(
                        rank_candidates_list, model=interest_model, language=language,
                    )
                    ranked_items = ranking_result.get("items") or []
            except Exception as exc:
                logger.warning("Extraction/ranking failed: %s", exc)
                warnings.append(f"Extraction/ranking failed: {exc}")
                ranked_items = []

        facts = ranked_items_to_facts(ranked_items or [], rank_candidates_list)

        if not facts:
            return api_error(
                "No rankable facts could be extracted.",
                code="NO_FACTS",
                status=422,
                hint="Provide ranked_candidates or a source_draft with extractable content.",
            )

        try:
            writer_result = write_script_from_facts(facts, language=language, target_blocks=target_blocks)
            script = writer_result["script"]
            llm_meta = writer_result.get("meta", {})
            llm_meta["status"] = "ok"
        except Exception as exc:
            logger.warning("Fact script writer failed; fallback to raw script: %s", exc)
            script = raw_script
            warnings.append(f"Fact script writer failed: {exc}")
            llm_meta = {
                "provider": "fact_writer",
                "status": "fallback_raw",
                "error": str(exc),
                "target_blocks": target_blocks,
            }

        lint_result = lint_script(script)

        if lint_result["status"] == "hard_fail":
            warnings.append(f"Script lint hard fail (score={lint_result['score']}), regenerating with strict style...")
            try:
                writer_result2 = write_script_from_facts(facts, language=language, target_blocks=target_blocks, style="strict")
                script2 = writer_result2["script"]
                lint_result2 = lint_script(script2)
                if lint_result2["score"] > lint_result["score"]:
                    script = script2
                    lint_result = lint_result2
                    llm_meta = writer_result2.get("meta", {})
                    llm_meta["status"] = "ok_after_hard_retry"
            except Exception:
                pass
        elif lint_result["status"] == "soft_fail":
            warnings.append(f"Script lint soft fail (score={lint_result['score']}), regenerating...")
            try:
                writer_result2 = write_script_from_facts(facts, language=language, target_blocks=target_blocks)
                script2 = writer_result2["script"]
                lint_result2 = lint_script(script2)
                if lint_result2["score"] > lint_result["score"]:
                    script = script2
                    lint_result = lint_result2
                    llm_meta = writer_result2.get("meta", {})
                    llm_meta["status"] = "ok_after_soft_retry"
            except Exception:
                pass

        if llm_meta:
            llm_meta["lint_result"] = lint_result
        else:
            llm_meta = {"lint_result": lint_result}

    section_count = len(source_draft.get("sections") or [])
    raw_block_count = len(raw_script.get("blocks") or [])
    output_block_count = len((script or {}).get("blocks") or [])

    response_payload = {
        "ok": True,
        "handling_mode": handling_mode,
        "source_draft": source_draft,
        "raw_draft_json": source_draft,
        "raw_script": raw_script,
        "script": script,
        "llm_meta": llm_meta,
        "warnings": warnings,
        "meta": {
            "section_count": section_count,
            "raw_block_count": raw_block_count,
            "output_block_count": output_block_count,
            "target_blocks": target_blocks,
        },
    }
    return jsonify(response_payload)


@app.route("/api/content/drafts/save", methods=["POST"])
def save_content_draft_file():
    data = request.get_json()
    if not isinstance(data, dict):
        return api_error(
            "Request body must be JSON.",
            code="INVALID_REQUEST_JSON",
            status=400,
            hint="Send source_draft and optional filename as JSON.",
        )

    source_draft_raw = data.get("source_draft")
    if source_draft_raw is None:
        raw_draft_json = data.get("raw_draft_json")
        if isinstance(raw_draft_json, dict):
            source_draft_raw = raw_draft_json
        elif isinstance(raw_draft_json, str) and raw_draft_json.strip():
            try:
                source_draft_raw = json.loads(raw_draft_json)
            except json.JSONDecodeError:
                return api_error(
                    "raw_draft_json is not valid JSON.",
                    code="INVALID_SOURCE_DRAFT_JSON",
                    status=400,
                    hint="Fix raw draft JSON syntax and retry.",
                )

    language = str(data.get("language") or "vi-VN").strip() or "vi-VN"
    try:
        source_draft = _normalize_source_draft_input(source_draft_raw, fallback_language=language)
    except ValueError as exc:
        return api_error(
            "source_draft is invalid.",
            code="INVALID_SOURCE_DRAFT",
            status=400,
            details=str(exc),
            hint="Fetch a source draft first or paste valid raw draft JSON.",
        )

    requested_filename = str(data.get("filename") or "").strip()
    if requested_filename:
        stem = Path(requested_filename).stem
        filename_stem = _sanitize_filename_token(stem)
    else:
        topic_token = _sanitize_filename_token(str(source_draft.get("topic_query") or "draft"))
        ts_token = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_stem = f"{topic_token}_{ts_token}"
    filename = f"{filename_stem}.json"

    path = DRAFTS_DIR / filename
    duplicate_index = 1
    while path.exists():
        path = DRAFTS_DIR / f"{filename_stem}_{duplicate_index}.json"
        duplicate_index += 1

    try:
        DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(source_draft, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.exception("Failed to save content draft file: %s", path)
        return api_error(
            "Failed to save content draft file.",
            code="SAVE_DRAFT_FAILED",
            status=500,
            details=str(exc),
            hint="Check write permissions and free disk space, then retry.",
        )

    return jsonify(
        {
            "ok": True,
            "filename": path.name,
            "saved_path": str(path.relative_to(PROJECT_ROOT)),
        }
    )


@app.route("/api/content/drafts", methods=["GET"])
def list_content_drafts():
    limit_raw = request.args.get("limit", "100")
    try:
        limit = int(limit_raw)
    except (TypeError, ValueError):
        return api_error(
            "limit must be an integer.",
            code="INVALID_PAGINATION",
            status=400,
            hint="Use a numeric limit between 1 and 500.",
        )
    if limit < 1 or limit > 500:
        return api_error(
            "limit out of range.",
            code="INVALID_PAGINATION",
            status=400,
            hint="Use a limit between 1 and 500.",
        )

    if not DRAFTS_DIR.exists():
        return jsonify({"ok": True, "drafts": [], "total": 0, "limit": limit})

    draft_files = sorted(DRAFTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    draft_items = []
    for path in draft_files[:limit]:
        stat = path.stat()
        created_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        draft_items.append(
            {
                "filename": path.name,
                "saved_path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": stat.st_size,
                "created_at": created_dt.isoformat(),
                "created_ts": stat.st_mtime,
            }
        )
    return jsonify(
        {
            "ok": True,
            "drafts": draft_items,
            "total": len(draft_files),
            "limit": limit,
        }
    )


@app.route("/api/content/drafts/<name>", methods=["GET"])
def get_content_draft(name):
    safe_name = Path(str(name or "")).name
    if not safe_name or safe_name in {".", ".."}:
        return api_error(
            "Draft name is required.",
            code="DRAFT_NAME_REQUIRED",
            status=400,
            hint="Select a saved draft and retry.",
        )

    if not safe_name.lower().endswith(".json"):
        safe_name = f"{safe_name}.json"

    path = DRAFTS_DIR / safe_name
    if not path.exists():
        return api_error(
            "Draft file not found.",
            code="DRAFT_NOT_FOUND",
            status=404,
            hint="Refresh saved drafts list and choose an existing file.",
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.exception("Failed to read draft file: %s", path)
        return api_error(
            "Draft file is invalid JSON.",
            code="INVALID_SOURCE_DRAFT_JSON",
            status=400,
            details=str(exc),
            hint="Open and fix the draft file JSON, then retry.",
        )

    try:
        source_draft = _normalize_source_draft_input(payload)
    except ValueError as exc:
        return api_error(
            "Draft file has invalid shape.",
            code="INVALID_SOURCE_DRAFT",
            status=400,
            details=str(exc),
            hint="Use a source_draft JSON generated by this app.",
        )

    stat = path.stat()
    created_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return jsonify(
        {
            "ok": True,
            "filename": path.name,
            "saved_path": str(path.relative_to(PROJECT_ROOT)),
            "source_draft": source_draft,
            "meta": {
                "size_bytes": stat.st_size,
                "created_at": created_dt.isoformat(),
                "created_ts": stat.st_mtime,
            },
        }
    )


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    if not isinstance(data, dict):
        return api_error(
            "Request body must be JSON.",
            code="INVALID_REQUEST_JSON",
            status=400,
            hint="Send script_name and script as JSON.",
        )
    script_name = data.get("script_name", "untitled")
    output_name = data.get("output_name") or script_name
    script_data = data.get("script")
    request_meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    save_script = bool(data.get("save_script", True))

    if not script_data or not script_data.get("blocks"):
        return api_error(
            "Script must include at least one block.",
            code="INVALID_SCRIPT",
            status=400,
            hint="Add a block with non-empty text, then generate again.",
        )

    fact_ids = []
    seen_fact_ids = set()
    for block in script_data.get("blocks", []):
        if not isinstance(block, dict):
            continue
        fact_id = str(block.get("fact_id") or "").strip()
        if not fact_id or fact_id in seen_fact_ids:
            continue
        seen_fact_ids.add(fact_id)
        fact_ids.append(fact_id)

    # Persist script payload (either to saved scripts or temporary job script)
    if save_script:
        script_path = PROJECT_ROOT / "json_scripts" / f"{script_name}.json"
        cleanup_script = False
    else:
        script_path = PROJECT_ROOT / "tmp" / "job_scripts" / f"{output_name}_{int(time.time() * 1000)}.json"
        cleanup_script = True
    try:
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(json.dumps(script_data, ensure_ascii=False, indent=2))
    except Exception as exc:
        logger.exception("Failed to persist script before generation: %s", script_path)
        return api_error(
            "Unable to save script before generation.",
            code="SAVE_SCRIPT_FAILED",
            status=500,
            details=str(exc),
            hint="Check write permissions and disk space, then retry.",
        )

    # Create job
    job_id = f"{output_name}_{int(time.time())}"
    jobs[job_id] = {
        "status": "starting",
        "stage": "loading",
        "progress": "Queued...",
        "current_block": None,
        "total_blocks": len(script_data.get("blocks", [])),
        "meta": request_meta,
        "fact_ids": fact_ids,
        "output": None,
        "error": None,
    }

    # Run in background thread
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, script_path, output_name, cleanup_script, fact_ids),
    )
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


def _run_job(job_id, script_path, output_name, cleanup_script=False, fact_ids=None):
    mgr = None

    def update_job(**fields):
        job = jobs.get(job_id)
        if not job:
            return
        job.update(fields)

    try:
        update_job(
            status="loading",
            stage="loading",
            progress="Loading models...",
            current_block=0,
        )
        mgr = create_manager()

        def on_progress(progress_event: dict):
            update_job(
                status="processing",
                stage=progress_event.get("stage", "rendering"),
                progress=progress_event.get("message", "Processing..."),
                current_block=progress_event.get("current_block"),
                total_blocks=progress_event.get("total_blocks"),
            )

        output = mgr.process_script(script_path, output_name, progress_callback=on_progress)

        total_blocks = jobs.get(job_id, {}).get("total_blocks")
        used_fact_count = 0
        if fact_ids:
            try:
                used_fact_count = CONTENT_BANK_STORE.mark_facts_used(
                    fact_ids,
                    video_id=output.name,
                )
            except Exception:
                logger.exception("Failed to mark fact cards as used for job %s.", job_id)
        update_job(
            status="done",
            stage="done",
            output=output.name,
            progress="Complete!",
            current_block=total_blocks,
            used_fact_count=used_fact_count,
        )
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        failed_stage = jobs.get(job_id, {}).get("stage") or "rendering"
        update_job(
            status="error",
            stage=failed_stage,
            progress="Failed to generate video.",
            error={
                "code": "GENERATION_FAILED",
                "message": "Video generation failed.",
                "details": str(e),
                "hint": "Check background/image paths, voice settings, and block text, then retry.",
            },
        )
    finally:
        if mgr:
            mgr.close()
        if cleanup_script and script_path.exists():
            script_path.unlink(missing_ok=True)


@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return api_error(
            "Job not found.",
            code="JOB_NOT_FOUND",
            status=404,
            hint="Start a new generation and poll the returned job ID.",
        )
    return jsonify(job)


@app.route("/output/<filename>")
def serve_output(filename):
    download_flag = request.args.get("download", "").strip().lower()
    as_attachment = download_flag in {"1", "true", "yes"}
    return send_from_directory(str(OUTPUT_DIR), filename, as_attachment=as_attachment)


@app.route("/api/outputs")
def list_outputs():
    outputs = sorted(OUTPUT_DIR.glob("*.mp4")) if OUTPUT_DIR.exists() else []
    payload = [output_to_payload(p) for p in outputs]
    payload.sort(key=lambda x: x.get("created_ts", 0), reverse=True)
    return jsonify(payload)


@app.route("/api/outputs/<filename>", methods=["DELETE"])
def delete_output(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return api_error(
            "Output file not found.",
            code="OUTPUT_NOT_FOUND",
            status=404,
            hint="Refresh the output list and retry.",
        )
    try:
        path.unlink()
        sibling_mp3 = path.with_suffix(".mp3")
        if sibling_mp3.exists():
            sibling_mp3.unlink()
        return jsonify({"ok": True})
    except Exception as exc:
        logger.exception("Failed to delete output: %s", path)
        return api_error(
            "Failed to delete output file.",
            code="DELETE_OUTPUT_FAILED",
            status=500,
            details=str(exc),
            hint="Close any media player using the file, then retry.",
        )


@app.route("/api/pipeline/auto", methods=["POST"])
def auto_pipeline():
    """Fully automated: topic -> fetch -> rank -> write -> lint -> generate video."""
    data = request.get_json()
    if not isinstance(data, dict):
        return api_error("Request body must be JSON.", code="INVALID_REQUEST_JSON", status=400)

    topic = str(data.get("topic") or "").strip()
    if not topic:
        return api_error("Topic is required.", code="TOPIC_REQUIRED", status=400)

    language = str(data.get("language") or "en-US").strip() or "en-US"
    target_blocks = int(data.get("target_blocks", 6))
    output_name = str(data.get("output_name") or "").strip() or _sanitize_filename_token(topic)

    warnings: list[str] = []

    # Step 1: Fetch from Wikipedia
    try:
        raw_result = fetch_wikipedia_draft(topic=topic, language_code=language, max_blocks=12)
    except Exception as exc:
        return api_error("Failed to fetch content.", code="SOURCE_FETCH_FAILED", status=502, details=str(exc))

    source_draft = raw_result.get("source_draft") or {}
    warnings.extend(raw_result.get("warnings") or [])

    # Step 2: Extract and rank source units
    extraction_result = extract_source_units_from_draft(source_draft, keep_top_k=24)
    top_units = extraction_result.get("top_units") or []

    ranked_items = []
    rank_candidates_list = []
    if top_units:
        interest_model = os.getenv("GROQ_INTEREST_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
        try:
            rank_candidates_list = to_rank_candidates(top_units)
            ranking_result = rank_interest_candidates(
                rank_candidates_list, model=interest_model, language=language,
            )
            ranked_items = ranking_result.get("items") or []
        except Exception as exc:
            warnings.append(f"Ranking failed, using local scores: {exc}")
            # Fallback: build facts directly from units
            rank_candidates_list = to_rank_candidates(top_units)

    # Step 3: Convert to facts and write script
    facts = ranked_items_to_facts(ranked_items, rank_candidates_list)
    if not facts and top_units:
        # Ultimate fallback: use raw unit text
        for u in top_units[:target_blocks * 2]:
            text = u.get("text", "") if isinstance(u, dict) else ""
            if text:
                facts.append({"fact_text": text})

    if not facts:
        return api_error("No rankable facts found for this topic.", code="NO_FACTS", status=422)

    try:
        writer_result = write_script_from_facts(facts, language=language, target_blocks=target_blocks)
        script = writer_result["script"]
        writer_meta = writer_result.get("meta", {})
    except Exception as exc:
        warnings.append(f"Fact writer failed: {exc}")
        script = {"language": language, "blocks": [{"text": f["fact_text"]} for f in facts[:target_blocks]]}
        writer_meta = {"status": "fallback", "error": str(exc)}

    # Step 4: Lint the script
    lint_result = lint_script(script)

    if lint_result["status"] == "soft_fail":
        warnings.append(f"Script lint soft fail (score={lint_result['score']}), regenerating...")
        try:
            writer_result2 = write_script_from_facts(facts, language=language, target_blocks=target_blocks, style="strict")
            script2 = writer_result2["script"]
            lint_result2 = lint_script(script2)
            if lint_result2["score"] > lint_result["score"]:
                script = script2
                lint_result = lint_result2
                writer_meta = writer_result2.get("meta", {})
        except Exception:
            pass

    if lint_result["status"] == "hard_fail":
        return api_error(
            f"Script quality too low (score={lint_result['score']}). Pipeline rejected.",
            code="QUALITY_HARD_FAIL",
            status=422,
            details=json.dumps(lint_result),
            hint="Try a different topic or adjust target_blocks.",
        )

    # Step 5: Generate video
    script_path = PROJECT_ROOT / "tmp" / "job_scripts" / f"{output_name}_{int(time.time() * 1000)}.json"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(json.dumps(script, ensure_ascii=False, indent=2))

    job_id = f"{output_name}_{int(time.time())}"
    jobs[job_id] = {
        "status": "starting",
        "stage": "loading",
        "progress": "Queued...",
        "current_block": None,
        "total_blocks": len(script.get("blocks", [])),
        "meta": {"pipeline": "auto", "topic": topic, "lint_score": lint_result["score"]},
        "fact_ids": [],
        "output": None,
        "error": None,
    }

    thread = threading.Thread(target=_run_job, args=(job_id, script_path, output_name, True, []))
    thread.daemon = True
    thread.start()

    return jsonify({
        "ok": True,
        "job_id": job_id,
        "script": script,
        "lint_result": lint_result,
        "writer_meta": writer_meta,
        "ranked_facts_count": len(facts),
        "warnings": warnings,
    })


@app.route("/api/agent/run", methods=["POST"])
def agent_run():
    """Agent mode: prompt -> autonomous video generation."""
    data = request.get_json()
    if not isinstance(data, dict):
        return api_error("Request body must be JSON.", code="INVALID_REQUEST_JSON", status=400)

    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        return api_error("Prompt is required.", code="PROMPT_REQUIRED", status=400)

    output_name = str(data.get("output_name") or "").strip()

    # Build explicit user config from request params
    from .agent.models import AgentConfig
    user_config = AgentConfig(
        image_display=data.get("image_display") or None,
        target_blocks=int(data["target_blocks"]) if data.get("target_blocks") else None,
        style=data.get("style") or None,
        voice=data.get("voice") or None,
    )

    job_id = f"agent_{int(time.time())}_{hash(prompt) % 10000:04d}"
    jobs[job_id] = {
        "status": "starting",
        "stage": "plan",
        "progress": "Starting agent...",
        "current_block": None,
        "total_blocks": None,
        "meta": {"pipeline": "agent", "prompt": prompt[:200]},
        "fact_ids": [],
        "output": None,
        "error": None,
    }

    def run_agent_job():
        from .agent import VideoAgent

        def on_progress(event: dict):
            job = jobs.get(job_id)
            if not job:
                return
            job.update({
                "status": "processing",
                "stage": event.get("phase") or event.get("stage", "processing"),
                "progress": event.get("message", "Processing..."),
                "current_block": event.get("current_block"),
                "total_blocks": event.get("total_blocks"),
            })

        agent = VideoAgent(progress_callback=on_progress)
        result = agent.run(prompt, output_name=output_name, user_config=user_config)

        job = jobs.get(job_id)
        if not job:
            return

        if result.success:
            job.update({
                "status": "done",
                "stage": "done",
                "progress": "Video generation complete!",
                "output": result.video_path,
                "total_blocks": result.metadata.get("actual_blocks"),
                "meta": {
                    **job.get("meta", {}),
                    "agent_result": {
                        "lint_score": result.lint_score,
                        "sources_used": result.sources_used,
                        "images_matched": result.images_matched,
                        "warnings": result.warnings,
                    },
                },
            })
        else:
            job.update({
                "status": "error",
                "stage": "failed",
                "progress": result.error or "Agent pipeline failed.",
                "error": result.error,
            })

    thread = threading.Thread(target=run_agent_job, daemon=True)
    thread.start()

    return jsonify({
        "ok": True,
        "job_id": job_id,
        "message": f"Agent started for: {prompt[:100]}",
    })


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("\n  Video Maker Web UI")
    print("  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
