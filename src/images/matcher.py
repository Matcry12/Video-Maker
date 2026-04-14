"""Image-to-text matching using SigLIP embeddings or Gemma4:e2b via Ollama."""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:e2b"


# ---------------------------------------------------------------------------
# SigLIP Mode
# ---------------------------------------------------------------------------

_siglip_model = None
_siglip_processor = None


def _load_siglip() -> None:
    """Lazy-load SigLIP model. Downloads ~350 MB on first use."""
    global _siglip_model, _siglip_processor
    if _siglip_model is not None:
        return
    try:
        import torch
        from transformers import AutoModel, AutoProcessor

        MODEL_ID = "google/siglip2-base-patch16-256"
        logger.info(
            "Loading SigLIP model: %s (first load downloads ~350 MB)...", MODEL_ID
        )
        _siglip_model = AutoModel.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32
        ).eval()
        _siglip_processor = AutoProcessor.from_pretrained(MODEL_ID)
        logger.info("SigLIP model loaded.")
    except Exception as exc:
        logger.error("Failed to load SigLIP: %s", exc)
        raise


def embed_texts_siglip(texts: list[str]) -> np.ndarray:
    """Embed texts with SigLIP. Returns (N, D) normalized float32 array."""
    _load_siglip()
    import torch

    inputs = _siglip_processor(
        text=texts, padding="max_length", return_tensors="pt"
    )
    with torch.no_grad():
        feats = _siglip_model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def embed_images_siglip(image_paths: list[Path]) -> np.ndarray:
    """Embed images with SigLIP. Returns (N, D) normalized float32 array."""
    _load_siglip()
    import torch
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = _siglip_processor(images=images, return_tensors="pt")
    with torch.no_grad():
        feats = _siglip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def match_images_siglip(
    block_texts: list[str],
    candidate_paths: list[Path],
    top_n: int = 3,
) -> dict[int, list[Path]]:
    """Match blocks to images using SigLIP cosine similarity.

    Returns {block_index: [image_paths]} with up to top_n images per block,
    sorted by descending similarity score.
    """
    if not block_texts or not candidate_paths:
        return {}

    text_embs = embed_texts_siglip(block_texts)       # (T, D)
    img_embs = embed_images_siglip(candidate_paths)   # (I, D)

    # Cosine similarity (both sides already L2-normalised)
    sim = text_embs @ img_embs.T  # (T, I)

    result: dict[int, list[Path]] = {}
    for t in range(len(block_texts)):
        scores = sim[t]
        # Get indices sorted by descending score
        ranked = np.argsort(-scores)[:top_n]
        result[t] = [candidate_paths[int(i)] for i in ranked]

    return result


# ---------------------------------------------------------------------------
# Ollama / Gemma4 Mode
# ---------------------------------------------------------------------------

def classify_image_ollama(
    image_path: Path,
    prompt: str = "Describe this image in 2-3 keywords.",
    model: str = "",
) -> str:
    """Classify/describe an image using Gemma4:e2b via Ollama.

    Follows the urllib pattern from interest_ranker.py.  Returns a description
    string, or empty string on failure.
    """
    model = model or os.getenv("OLLAMA_IMAGE_MODEL", "").strip() or DEFAULT_OLLAMA_MODEL
    base_url = (os.getenv("OLLAMA_BASE_URL", "").strip() or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    endpoint = f"{base_url}/api/chat"

    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
    except OSError as exc:
        logger.warning("Cannot read image %s: %s", image_path, exc)
        return ""

    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 100},
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
    }

    req = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=60.0) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            body = json.loads(resp.read().decode(charset, errors="replace"))
        return str(body.get("message", {}).get("content", "")).strip()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        logger.warning("Ollama HTTP error %s for %s: %s", exc.code, image_path.name, detail)
    except (URLError, TimeoutError) as exc:
        logger.warning("Ollama network error for %s: %s", image_path.name, exc)
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Ollama response parse error for %s: %s", image_path.name, exc)
    return ""


def match_images_ollama(
    block_texts: list[str],
    candidate_paths: list[Path],
    top_n: int = 3,
) -> dict[int, list[Path]]:
    """Match images to blocks using Ollama classification + word overlap.

    Returns {block_index: [image_paths]} with up to top_n per block.
    """
    if not block_texts or not candidate_paths:
        return {}

    # Classify each image once
    descriptions: list[str] = []
    for path in candidate_paths:
        desc = classify_image_ollama(path)
        descriptions.append(desc.lower())

    result: dict[int, list[Path]] = {}
    for t_idx, block_text in enumerate(block_texts):
        block_words = set(block_text.lower().split())
        scored: list[tuple[int, int]] = []
        for i_idx, desc in enumerate(descriptions):
            desc_words = set(desc.split())
            overlap = len(block_words & desc_words)
            scored.append((overlap, i_idx))
        scored.sort(key=lambda x: -x[0])
        result[t_idx] = [candidate_paths[i] for _, i in scored[:top_n]]

    return result


# ---------------------------------------------------------------------------
# Unified Interface
# ---------------------------------------------------------------------------

def match_images_roundrobin(
    block_texts: list[str],
    candidate_paths: list[Path],
) -> dict[int, list[Path]]:
    """Assign images to blocks in round-robin order. No ML model required.

    Distributes ALL candidates across blocks so each block gets multiple images.
    """
    if not block_texts or not candidate_paths:
        return {}
    n_blocks = len(block_texts)
    result: dict[int, list[Path]] = {i: [] for i in range(n_blocks)}
    for ci, cpath in enumerate(candidate_paths):
        block_idx = ci % n_blocks
        result[block_idx].append(cpath)
    # Ensure every block has at least one image (cycle if needed)
    for i in range(n_blocks):
        if not result[i]:
            result[i] = [candidate_paths[i % len(candidate_paths)]]
    return result


def match_images_to_blocks(
    block_texts: list[str],
    candidate_paths: list[Path],
    mode: str = "",
) -> dict[int, list[Path]]:
    """Match images to script blocks using the configured mode.

    mode: "roundrobin" (default, no model), "siglip", "ollama", or "none".
    Read from env IMAGE_MATCH_MODE if not specified.

    Returns {block_index: [image_paths]} — multiple images per block.
    """
    if not block_texts or not candidate_paths:
        return {}

    resolved_mode = (
        mode
        or os.getenv("IMAGE_MATCH_MODE", "").strip().lower()
        or "roundrobin"
    )

    if resolved_mode == "none":
        return {}

    if resolved_mode == "ollama":
        return match_images_ollama(block_texts, candidate_paths)

    if resolved_mode == "siglip":
        try:
            return match_images_siglip(block_texts, candidate_paths)
        except Exception as exc:
            logger.warning("SigLIP matching failed (%s). Falling back to roundrobin.", exc)
            return match_images_roundrobin(block_texts, candidate_paths)

    # Default: roundrobin (no model loading, no OOM risk)
    return match_images_roundrobin(block_texts, candidate_paths)
