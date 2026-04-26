"""Gaming trend discovery — fetches trending signals from Reddit r/gaming and r/Games."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

TIMEOUT = 10

_STRIP_FLAIR_RE = re.compile(r"^\[[^\]]+\]\s*")
_PUNCT_RE = re.compile(r"[^\w\s]")

_NON_GAME_TOKENS = frozenset({
    "meme", "art", "cosplay", "irl", "photo", "screenshot",
})

_REDDIT_HEADERS = {"User-Agent": "VideoMaker/1.0"}


def _fetch_subreddit_hot(subreddit: str) -> list[dict[str, Any]]:
    """Fetch hot posts from a subreddit using Reddit JSON API. Never raises."""
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
    req = urllib.request.Request(url, headers=_REDDIT_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    posts = []
    for child in (data.get("data") or {}).get("children") or []:
        post = (child.get("data") or {})
        if not post:
            continue

        title: str = post.get("title") or ""
        title = _STRIP_FLAIR_RE.sub("", title).strip()

        # Skip obvious non-game posts
        title_lower = title.lower()
        if any(token in title_lower for token in _NON_GAME_TOKENS):
            continue

        selftext: str = post.get("selftext") or ""
        score: int = int(post.get("score") or 0)
        num_comments: int = int(post.get("num_comments") or 0)

        posts.append({
            "title": title,
            "selftext": selftext[:300],
            "score": score,
            "num_comments": num_comments,
            "source": f"reddit_{subreddit.lower()}",
        })

    return posts


def _fetch_reddit_gaming() -> list[dict[str, Any]]:
    """Fetch hot posts from r/gaming."""
    return _fetch_subreddit_hot("gaming")


def _fetch_reddit_games() -> list[dict[str, Any]]:
    """Fetch hot posts from r/Games (more discussion-heavy, higher quality)."""
    return _fetch_subreddit_hot("Games")


def fetch_trending_games(limit: int = 10) -> list[dict]:
    """Fetch trending games from Reddit.

    Returns list of candidate dicts with keys:
      title, trending_reason, reddit_context, score, urgency_score,
      sources (list[str]), genres (list), anilist_synopsis (str)
    Sorted by score descending. Never raises — returns [] on total failure.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_fetch_reddit_gaming): "reddit_gaming",
            executor.submit(_fetch_reddit_games): "reddit_games",
        }
        raw_results: list[dict[str, Any]] = []
        for future in as_completed(futures):
            try:
                raw_results.extend(future.result())
            except Exception:
                pass

    # Deduplicate by normalized title (keep highest score)
    seen: dict[str, dict[str, Any]] = {}
    for post in raw_results:
        norm = _PUNCT_RE.sub("", post["title"].lower()).strip()
        if norm not in seen or post["score"] > seen[norm]["score"]:
            seen[norm] = post

    candidates: list[dict] = []
    for post in seen.values():
        reddit_score = post["score"]
        num_comments = post["num_comments"]
        raw_score = (reddit_score / 50000) * 0.6 + (num_comments / 500) * 0.4
        score = min(1.0, raw_score)

        candidates.append({
            "title": post["title"],
            "trending_reason": f"Trending on r/{post['source'].replace('reddit_', '')} with {reddit_score:,} upvotes and {num_comments:,} comments",
            "reddit_context": post["selftext"],
            "score": score,
            "urgency_score": 0.5,
            "sources": [post["source"]],
            "genres": [],
            "anilist_synopsis": post["selftext"],
            "_category": "gaming",
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:limit]
