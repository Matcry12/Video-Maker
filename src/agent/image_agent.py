"""Image agent — searches, downloads, and matches images to script blocks."""

import gc
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from .models import AgentPlan, ImageResult

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_images(
    script: dict[str, Any],
    plan: AgentPlan,
    emit: Optional[Callable[[dict], None]] = None,
) -> ImageResult:
    """Find and match images for each script block.

    Uses image_keywords from script blocks when available, falls back to
    text-based keyword extraction.
    Ensures every block has at least one image — runs fallback searches for empty blocks.
    Never raises — returns ImageResult with unmodified script on failure.
    """
    warnings: list[str] = []

    def _emit(phase: str, message: str, **extra):
        if emit:
            try:
                emit({"phase": phase, "message": message, **extra})
            except Exception:
                pass

    _emit("image", "Finding relevant images...")

    image_map: dict[int, list[str]] = {}
    total_images = 0

    try:
        from ..images.pipeline import get_images_for_script

        # Fetch enough images to cover the entire video
        # At 6-8s per image, a 2-minute video needs ~15-20 images
        # Fetch as many as possible — better to have extras than gaps
        raw_map = get_images_for_script(
            script, plan.topic,
            topic_category=getattr(plan, 'topic_category', ''),
            images_per_block=30,
        )
        for idx, img_entries in raw_map.items():
            if 0 <= idx < len(script.get("blocks", [])):
                rel_paths: list[str] = []
                for entry in img_entries:
                    # Support both dict {"path": Path, "keyword": str} and plain Path
                    img_path = entry["path"] if isinstance(entry, dict) else entry
                    rel_path = str(img_path)
                    try:
                        rel_path = str(Path(img_path).relative_to(PROJECT_ROOT))
                    except (ValueError, TypeError):
                        pass
                    rel_paths.append(rel_path)
                script["blocks"][idx]["image"] = rel_paths
                image_map[idx] = rel_paths
                total_images += len(rel_paths)

        # Check for blocks with no images and warn
        blocks = script.get("blocks", [])
        empty_blocks = [i for i in range(len(blocks)) if i not in image_map]
        if empty_blocks:
            warnings.append(
                f"Blocks with no images after all fallbacks: {empty_blocks}. "
                "These blocks will show the default background video."
            )
            logger.warning(
                "Image agent: %d/%d blocks have no images: %s",
                len(empty_blocks), len(blocks), empty_blocks,
            )

        logger.info(
            "Matched %d images to %d/%d blocks (%d total).",
            len(image_map), len(blocks) - len(empty_blocks),
            len(blocks), total_images,
        )
    except Exception as exc:
        warnings.append(
            f"Image pipeline failed: {exc}. Video will use default backgrounds."
        )
        logger.warning("Image pipeline failed: %s", exc)

    gc.collect()

    return ImageResult(
        script=script,
        image_map=image_map,
        total_images=total_images,
        warnings=warnings,
    )
