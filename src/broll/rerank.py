"""SigLIP clip reranker for the b-roll pipeline.

Public API:
    rerank_pool(candidates, work_dir, *, max_candidates, frames_per_clip)
        -> Optional[list[dict]]

Each candidate is scored against its OWN query (candidates in one beat pool
can come from different queries). Frames are downloaded in parallel and
embedded in a single batch. On any failure (model load, downloads, torch)
the function logs a warning and returns None so the caller can fall back to
the unranked pool.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_THUMB_WORKERS = 8


# ---------------------------------------------------------------------------
# Internal helpers (mirror of lab/broll_rerank_ab.py — no src imports at
# module level so a torch failure never breaks the broll package import)
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"[^\w]+", "_", text.strip().lower())[:60]


def _download_frame(url: str, dest: Path) -> Optional[Path]:
    """Download a single frame thumbnail. Returns dest on success, None on failure."""
    import requests  # local import — only needed at rerank time
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest
    except Exception as exc:
        logger.debug("Frame download failed %s: %s", url, exc)
        return None


def _sample_frames(
    video: dict,
    thumb_root: Path,
    frames: int,
) -> list[tuple[int, Path]]:
    """Download up to `frames` evenly-spaced thumbnails for a clip.

    Returns list of (nr, local_path) for successfully downloaded frames.
    Falls back to [image] poster if video_pictures is empty.
    """
    vid_id = video["id"]
    kslug = _slug(video.get("query", str(vid_id)))
    pictures = video.get("video_pictures", [])

    if pictures:
        n = len(pictures)
        if n <= frames:
            chosen = pictures
        else:
            step = n / frames
            chosen = [pictures[int(i * step)] for i in range(frames)]
    else:
        poster_url = video.get("image", "")
        if not poster_url:
            return []
        chosen = [{"nr": 0, "picture": poster_url}]

    jobs: list[tuple[int, str, Path]] = []
    for pic in chosen:
        nr = int(pic.get("nr", 0))
        url = pic.get("picture", "")
        if not url:
            continue
        dest = thumb_root / kslug / f"{vid_id}_{nr}.jpg"
        jobs.append((nr, url, dest))

    results: list[tuple[int, Path]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=_THUMB_WORKERS) as pool:
        futures = {
            pool.submit(_download_frame, url, dest): (nr, dest)
            for nr, url, dest in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            nr, dest = futures[fut]
            path = fut.result()
            if path is not None:
                results.append((nr, path))

    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank_pool(
    candidates: list[dict],
    work_dir: Path,
    *,
    max_candidates: int,
    frames_per_clip: int,
) -> Optional[list[dict]]:
    """Score and rerank `candidates` using SigLIP frame embeddings.

    Each candidate is scored against its own "query" field (plan decision C:
    candidates in one beat pool may come from different queries, so per-clip
    query semantics are preserved).

    Args:
        candidates:     List of candidate dicts with keys:
                        id, duration, best_url, w, h, query, image,
                        video_pictures.  Input is expected to be
                        resolution-sorted (best first); trimmed to
                        max_candidates before scoring.
        work_dir:       Beat work directory; thumbnails go in
                        work_dir/rerank_thumbs/.
        max_candidates: Hard cap on how many clips are scored.
        frames_per_clip: Number of evenly-spaced frames to sample per clip.

    Returns:
        Candidates list sorted by rerank_score desc, each annotated with
        "rerank_score" (float) and "best_offset" (float, seconds).
        Returns None on any failure so the caller can fall back gracefully.
    """
    try:
        # Lazy imports — keeps torch failures from breaking module import.
        import numpy as np
        from src.images.matcher import embed_texts_siglip, embed_images_siglip
    except Exception as exc:
        logger.warning("rerank_pool: failed to import SigLIP embedders (%s); skipping", exc)
        return None

    try:
        pool = candidates[:max_candidates]
        if not pool:
            return None

        thumb_root = work_dir / "rerank_thumbs"
        thumb_root.mkdir(parents=True, exist_ok=True)

        # --- 1. Embed unique query strings (one batch) -----------------------
        unique_queries = list(dict.fromkeys(c.get("query", "") for c in pool))
        try:
            text_embs = embed_texts_siglip(unique_queries)   # (Q, D)
        except Exception as exc:
            logger.warning("rerank_pool: text embedding failed (%s); skipping", exc)
            return None

        query_idx = {q: i for i, q in enumerate(unique_queries)}

        # --- 2. Download all frames in parallel across all clips --------------
        for cand in pool:
            cand["_sampled"] = _sample_frames(cand, thumb_root, frames_per_clip)

        # --- 3. Collect all frame paths for a single image-embed batch --------
        all_paths: list[Path] = []
        frame_map: list[tuple[int, int]] = []  # (clip_idx, frame_pos_within_clip)
        for ci, cand in enumerate(pool):
            for fi, (_nr, path) in enumerate(cand["_sampled"]):
                all_paths.append(path)
                frame_map.append((ci, fi))

        if not all_paths:
            logger.warning("rerank_pool: no frames downloaded for any candidate; skipping")
            for cand in pool:
                cand.pop("_sampled", None)
                cand["rerank_score"] = 0.0
                cand["best_offset"] = 0.0
            return sorted(pool, key=lambda c: c["rerank_score"], reverse=True)

        try:
            img_embs = embed_images_siglip(all_paths)   # (N, D)
        except Exception as exc:
            logger.warning("rerank_pool: image embedding failed (%s); skipping", exc)
            for cand in pool:
                cand.pop("_sampled", None)
            return None

        # --- 4. Compute cosine sims (both embeddings are L2-normalized) -------
        scores_flat: np.ndarray = (img_embs @ text_embs.T)   # (N, Q)

        # --- 5. Max-pool per clip against its own query ----------------------
        clip_frame_scores: list[list[tuple[int, float]]] = [[] for _ in pool]
        for flat_idx, (ci, fi) in enumerate(frame_map):
            nr = pool[ci]["_sampled"][fi][0]
            q = pool[ci].get("query", "")
            qi = query_idx.get(q, 0)
            sim = float(scores_flat[flat_idx, qi])
            clip_frame_scores[ci].append((nr, sim))

        for ci, cand in enumerate(pool):
            frame_scores = clip_frame_scores[ci]
            if not frame_scores:
                cand["rerank_score"] = 0.0
                cand["best_offset"] = 0.0
            else:
                best_nr, best_score = max(frame_scores, key=lambda x: x[1])
                cand["rerank_score"] = best_score
                max_nr = max(nr for nr, _ in frame_scores)
                if max_nr > 0:
                    cand["best_offset"] = round(
                        (best_nr / max_nr) * float(cand["duration"]), 1
                    )
                else:
                    cand["best_offset"] = 0.0
            cand.pop("_sampled", None)

        pool.sort(key=lambda c: c["rerank_score"], reverse=True)
        return pool

    except Exception as exc:
        logger.warning("rerank_pool: unexpected error (%s); falling back", exc)
        return None
