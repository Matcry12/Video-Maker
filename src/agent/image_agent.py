"""Image agent — searches, downloads, and matches images to script blocks."""

import gc
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

from ..agent_config import load_agent_settings
from ..images.keyword_enrichment import enrich_keywords
from ..images.phrase_windows import _STOPWORD_PROPERS
from .entity_sanitizer import forbidden_entities, sanitize_list
from .models import AgentPlan, ImageResult

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

_IMAGES_PER_BLOCK = int(load_agent_settings().get("images_per_block", 30))


def run_images(
    script: dict[str, Any],
    plan: AgentPlan,
    emit: Optional[Callable[[dict], None]] = None,
) -> ImageResult:
    """Find and match images for each script block.

    Filters contaminated keywords (foreign franchise names) before searching.
    Falls back to topic-anchored keywords for empty or fully-contaminated blocks.
    Never raises — returns ImageResult with unmodified script on failure.
    """
    emit = emit or (lambda _e: None)
    warnings: list[str] = []
    dropped_map: dict[int, list[str]] = {}

    def _emit(phase: str, message: str, **extra):
        try:
            emit({"phase": phase, "message": message, **extra})
        except Exception:
            pass

    _emit("image", "Finding relevant images...")

    image_map: dict[int, list[str]] = {}

    try:
        from ..images.pipeline import get_images_for_script

        fb = forbidden_entities(
            getattr(plan, "topic", ""),
            topic_aliases=list(getattr(plan, "entity_aliases", []) or []),
        )

        blocks = script.get("blocks") or []

        for i, block in enumerate(blocks):
            raw_keywords = list(block.get("image_keywords") or [])
            if not raw_keywords:
                raw_keywords = _synthesize_keywords(plan, block, i)

            enriched = enrich_keywords(raw_keywords, plan.topic, getattr(plan, "topic_category", ""))
            kept, dropped = sanitize_list(enriched, fb)

            if dropped:
                dropped_map[i] = dropped
                warnings.append(
                    f"Block {i}: dropped {len(dropped)} contaminated keywords "
                    f"({', '.join(dropped[:2])}{'...' if len(dropped) > 2 else ''})"
                )
            if not kept:
                kept = _synthesize_keywords(plan, block, i)
                warnings.append(f"Block {i}: all keywords were contaminated; using topic defaults")

            block["image_keywords"] = kept

        raw_map = get_images_for_script(
            script=script, topic=plan.topic,
            topic_category=getattr(plan, "topic_category", ""),
            images_per_block=_IMAGES_PER_BLOCK,
        )

        for idx, img_entries in (raw_map or {}).items():
            if not (0 <= idx < len(blocks)):
                continue
            paths = []
            for entry in img_entries:
                p = entry.get("path") if isinstance(entry, dict) else entry
                if not p:
                    continue
                try:
                    p = str(Path(p).relative_to(PROJECT_ROOT))
                except (ValueError, TypeError):
                    pass
                paths.append(str(p))
            if paths:
                blocks[idx]["image"] = paths
                image_map[idx] = paths

        empty = [i for i in range(len(blocks)) if i not in image_map]
        if empty:
            for i in empty:
                blocks[i]["image_keywords"] = _synthesize_keywords(plan, blocks[i], i)
            only_empty_script = {
                "blocks": [
                    (blocks[i] if i in empty else {"text": "", "image_keywords": []})
                    for i in range(len(blocks))
                ]
            }
            second = get_images_for_script(
                script=only_empty_script, topic=plan.topic,
                topic_category=getattr(plan, "topic_category", ""),
                images_per_block=_IMAGES_PER_BLOCK,
            )
            for idx, entries in (second or {}).items():
                if idx not in empty:
                    continue
                paths = []
                for e in entries:
                    p = e.get("path") if isinstance(e, dict) else e
                    if not p:
                        continue
                    try:
                        p = str(Path(p).relative_to(PROJECT_ROOT))
                    except (ValueError, TypeError):
                        pass
                    paths.append(str(p))
                if paths:
                    blocks[idx]["image"] = paths
                    image_map[idx] = paths

        blocks_count = len(blocks)
        logger.info(
            "Matched %d images to %d/%d blocks.",
            len(image_map), len(image_map), blocks_count,
        )

    except Exception as exc:
        warnings.append(f"Image pipeline failed: {exc}. Video will use default backgrounds.")
        logger.warning("Image pipeline failed: %s", exc)

    gc.collect()

    still_empty = [i for i in range(len(script.get("blocks", []))) if i not in image_map]
    if still_empty:
        raise ValueError(
            f"No images found for block(s) {still_empty} after contamination filtering and "
            f"fallback search. Dropped keywords: {dropped_map}. Re-run with broader queries."
        )

    return ImageResult(
        script=script,
        image_map=image_map,
        total_images=sum(len(v) for v in image_map.values()),
        warnings=warnings,
    )


def _synthesize_keywords(plan, block, idx: int) -> list[str]:
    """Build generic topic-anchored keywords from the block text + topic."""
    topic = getattr(plan, "topic", "") or ""
    text = (block.get("text") or "").strip()
    proper = [w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", text) if w not in _STOPWORD_PROPERS][:3]
    seeds = [topic + " scene", topic + " portrait", topic + " official art"]
    for p in proper:
        seeds.append(f"{topic} {p}")
    seen = set()
    out = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:6]
