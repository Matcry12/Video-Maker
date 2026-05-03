"""Perceptual image deduplication using pHash."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_HASH_THRESHOLD = 8  # hamming distance — images within this are near-dupes


def dedup_images(paths: list[Path], threshold: int = _HASH_THRESHOLD) -> list[Path]:
    """Remove visually similar images using perceptual hashing.

    Keeps the first image when a near-duplicate is found.
    Falls back to returning all paths if imagehash is not installed.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        logger.debug("imagehash not installed; skipping perceptual dedup")
        return paths

    kept: list[tuple[Path, any]] = []
    for p in paths:
        try:
            h = imagehash.phash(Image.open(p).convert("RGB"))
        except Exception:
            kept.append((p, None))
            continue
        is_dupe = any(
            existing_h is not None and abs(h - existing_h) <= threshold
            for _, existing_h in kept
        )
        if not is_dupe:
            kept.append((p, h))

    dropped = len(paths) - len(kept)
    if dropped:
        logger.info("Perceptual dedup: removed %d near-duplicate image(s)", dropped)
    return [p for p, _ in kept]
