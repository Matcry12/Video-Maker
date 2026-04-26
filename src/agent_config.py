"""Profile config loader — models + agent runtime settings.

Reads profiles/default.json once, caches it. Exposes:
    - resolve_stage(stage): per-stage model + provider config
    - load_agent_settings(): tuning knobs (image density, TTS parallelism,
      research depth)

Models precedence (applied by callers/llm_client):
    1. explicit `model=` kwarg
    2. stage-specific override in profile (if stage was provided)
    3. profile defaults
    4. built-in defaults (below)
    5. GROQ_MODEL env var — only when no stage is given (ad-hoc callers)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROFILE_PATH = _PROJECT_ROOT / "profiles" / "default.json"

_BUILTIN_DEFAULTS: dict[str, object] = {
    "providers": ["gemini", "groq"],
    "gemini_model": "gemma-3-27b-it",
    "groq_model": "llama-3.3-70b-versatile",
}


@dataclass(frozen=True)
class StageModelCfg:
    providers: tuple[str, ...]
    gemini_model: str
    groq_model: str
    no_think: bool = False


_profile_cache: dict | None = None


def _load_profile() -> dict:
    """Load and cache the full profile. Missing file / parse error => {}."""
    global _profile_cache
    if _profile_cache is not None:
        return _profile_cache
    try:
        with open(_PROFILE_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.debug("Profile not found at %s — using built-in defaults.", _PROFILE_PATH)
        data = {}
    except Exception as exc:
        logger.warning("Failed to read profile %s: %s — using built-in defaults.", _PROFILE_PATH, exc)
        data = {}
    _profile_cache = data if isinstance(data, dict) else {}
    return _profile_cache


def _load_models_section() -> dict:
    """Return the 'models' section, or {}."""
    return _load_profile().get("models") or {}


def load_agent_settings() -> dict:
    """Return the 'agent' section (tuning knobs), or {}.

    Callers should use `.get(key, default)` with their own builtin fallback so
    the code still runs if the profile is missing or the key absent.
    """
    return _load_profile().get("agent") or {}


def resolve_stage(stage: str | None) -> StageModelCfg:
    """Return resolved config for a stage.

    Unknown or None stage => defaults only. Missing keys in a stage inherit
    from defaults.
    """
    models = _load_models_section()
    defaults = {**_BUILTIN_DEFAULTS, **(models.get("defaults") or {})}
    stage_cfg = (models.get("stages") or {}).get(stage or "", {})

    merged = {**defaults, **(stage_cfg or {})}
    providers = merged["providers"] or _BUILTIN_DEFAULTS["providers"]
    return StageModelCfg(
        providers=tuple(str(p).strip().lower() for p in providers if str(p).strip()),
        gemini_model=str(merged["gemini_model"]),
        groq_model=str(merged["groq_model"]),
        no_think=bool(merged.get("no_think", False)),
    )


def clear_cache() -> None:
    """Force reload on next call. For tests."""
    global _profile_cache
    _profile_cache = None


def research_settings(topic_category: str = "") -> dict:
    """Return the agent.research config dict with defaults applied.

    `topic_category` — when provided, overrides `require_min_tier` using the
    `tier_floor_by_category` map in the profile.  Categories like "anime" and
    "gaming" use tier 3 so fandom / fan-analysis sites aren't pre-filtered.
    """
    agent = load_agent_settings()
    cfg = dict(agent.get("research", {}))
    defaults = {
        "min_grounded_facts": 6,
        "min_tier_diversity": 2,
        "max_final_facts": 12,
        "max_research_iterations": 2,
        "tier_1_weight": 1.50,
        "tier_2_weight": 1.00,
        "tier_3_weight": 0.40,
        "tier_4_weight": 0.25,
        "max_ddg_per_query": 10,
        "max_crawl_pages": 8,
        "max_extract_pages": 8,
        "extract_concurrency": 3,
        "page_text_max_chars": 8000,
        "chunk_size_chars": 800,
        "chunk_min_chars": 150,
        "chunk_overlap_chars": 100,
        "parent_window_chunks": 2,
        "retrieval_top_k": 12,
        "bm25_threshold": 1.2,
        "grounding_fuzzy_threshold": 0.80,
        "rag_cache_ttl_secs": 7 * 24 * 3600,
        "rag_cache_rrf_k": 60,
        "n_query_expansion": 4,
        "require_min_tier": 2,
        "tier_floor_by_category": {},
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    if topic_category:
        tier_map: dict = cfg.get("tier_floor_by_category") or {}
        override = tier_map.get(topic_category.lower())
        if override is not None:
            cfg["require_min_tier"] = int(override)
    return cfg
