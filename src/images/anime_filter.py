"""Content-based anime image filter via deepghs imgutils models.

Replaces brittle domain blocklists. Classifies downloaded images and
drops anything that isn't an anime illustration / screenshot / comic.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def is_anime_image(path: str | Path) -> bool:
    """Return True if the image is anime-style art, False if real/3D/etc.

    Uses imgutils anime_real binary classifier. Lazy-imports so the module
    has no import cost when the filter is disabled.
    """
    try:
        from imgutils.validate import anime_real
        label, _score = anime_real(str(path))
        return label != "real"
    except Exception as exc:
        logger.warning("anime_filter: classifier failed on %s: %s — keeping by default", path, exc)
        return True


def filter_anime_images(paths: list[str | Path]) -> list[Path]:
    """Filter a list of image paths, keeping only anime-style ones."""
    kept: list[Path] = []
    for p in paths:
        if is_anime_image(p):
            kept.append(Path(p))
        else:
            logger.info("anime_filter: dropped non-anime image %s", p)
    return kept
