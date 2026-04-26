"""Anime trend discovery — fetches trending signals from AniList, Jikan, Reddit."""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

TIMEOUT = 10  # seconds per source

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_STRIP_EPISODE_RE = re.compile(r"-\s*[Ee]pisode\s+\d+.*$")
_STRIP_FLAIR_RE = re.compile(r"^\[[^\]]+\]\s*")
_PUNCT_RE = re.compile(r"[^\w\s]")

# Long-running daily shows that dominate AniList trending but aren't useful content targets.
# AniList doesn't tag these as "Kids" — they appear as Comedy/Slice of Life.
# Matched against normalized (lowercase, no punctuation) title.
_DAILY_SHOW_TOKENS = frozenset({
    "shin chan", "shinchan", "crayon shin",
    "anpanman",
    "sazae",
    "doraemon",
    "nintama",
    "chibi maruko",
    "hanakappa",
    "shimajiro",
    "ojarumaru", "mackaroo",
    "manga nihon mukashibanashi",
    "mahou no mako",
    "kirin no monoshiri",
})


def fetch_trending_anime(limit: int = 10) -> list[dict]:
    """Fetch and score trending anime from AniList + Jikan + Reddit.

    Returns list of dicts with keys:
      title, anilist_synopsis, genres, trending_reason, reddit_context,
      score (0-1), urgency_score (0-1), sources (list[str])
    Sorted by score descending. Never raises — returns [] on total failure.
    """
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_fetch_anilist): "anilist",
            executor.submit(_fetch_jikan): "jikan",
            executor.submit(_fetch_reddit): "reddit",
        }
        results: dict[str, list[dict]] = {}
        for future in as_completed(futures):
            source = futures[future]
            try:
                results[source] = future.result()
            except Exception:
                results[source] = []

    anilist_items = results.get("anilist", [])
    jikan_items = results.get("jikan", [])
    reddit_items = results.get("reddit", [])

    return _score_candidates(anilist_items, jikan_items, reddit_items, limit)


# ---------------------------------------------------------------------------
# Source fetchers
# ---------------------------------------------------------------------------

def _current_season() -> tuple[str, int]:
    """Return (SEASON_NAME, year) for the current anime season."""
    import datetime
    now = datetime.datetime.now()
    month = now.month
    year = now.year
    if month <= 3:
        return "WINTER", year
    elif month <= 6:
        return "SPRING", year
    elif month <= 9:
        return "SUMMER", year
    else:
        return "FALL", year


def _is_daily_show(title: str, next_ep_num: int) -> bool:
    """Return True for long-running daily children's shows to exclude."""
    norm = _PUNCT_RE.sub("", title.lower())
    for token in _DAILY_SHOW_TOKENS:
        if token in norm:
            return True
    return next_ep_num > 800


def _fetch_anilist() -> list[dict]:
    """Fetch top trending anime for the current season from AniList GraphQL."""
    season, year = _current_season()
    query = (
        f"query {{ Page(page:1,perPage:30) {{ media("
        f"season:{season},seasonYear:{year},"
        f"sort:TRENDING,type:ANIME,status:RELEASING,"
        f"format_in:[TV,ONA]) {{ "
        f"idMal title{{romaji english}} trending popularity "
        f"description(asHtml:false) genres episodes "
        f"nextAiringEpisode{{timeUntilAiring episode}} }} }} }}"
    )
    payload = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        "https://graphql.anilist.co",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "anime-trend-bot/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            data = json.loads(resp.read().decode(charset, errors="replace"))
    except Exception:
        return []

    media_list = (
        (data.get("data") or {})
        .get("Page", {})
        .get("media") or []
    )

    items: list[dict] = []
    for rank, media in enumerate(media_list, start=1):
        title_obj = media.get("title") or {}
        title = title_obj.get("english") or title_obj.get("romaji") or ""
        if not title:
            continue

        raw_desc = media.get("description") or ""
        synopsis = _HTML_TAG_RE.sub("", raw_desc).strip()

        next_ep = media.get("nextAiringEpisode") or {}
        next_ep_num = next_ep.get("episode") or 0

        if _is_daily_show(title, next_ep_num):
            continue

        episode_drop_hours: float | None = None
        if next_ep:
            seconds = next_ep.get("timeUntilAiring")
            if seconds is not None:
                episode_drop_hours = -(seconds / 3600.0)

        items.append({
            "title_romaji": title_obj.get("romaji") or "",
            "title_english": title_obj.get("english") or "",
            "title": title,
            "mal_id": media.get("idMal"),
            "trending_rank": rank,
            "popularity": media.get("popularity") or 0,
            "synopsis": synopsis,
            "genres": media.get("genres") or [],
            "episode_drop_hours": episode_drop_hours,
        })

    return items


def _fetch_jikan() -> list[dict]:
    """Fetch current-season anime from Jikan (MAL) to match AniList seasonal data."""
    import logging as _logging
    import socket
    _jikan_log = _logging.getLogger(__name__)

    season_name, year = _current_season()
    season_lower = season_name.lower()
    url = f"https://api.jikan.moe/v4/seasons/{year}/{season_lower}?limit=25"
    headers = {"User-Agent": "anime-trend-bot/1.0"}

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                data = json.loads(resp.read().decode(charset, errors="replace"))
            break
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            if attempt == 0:
                _jikan_log.warning("_fetch_jikan: attempt 1 failed (%s), retrying in 2s", exc)
                time.sleep(2)
                continue
            _jikan_log.warning("_fetch_jikan: both attempts failed: %s", exc)
            return []
        except Exception as exc:
            _jikan_log.warning("_fetch_jikan: unexpected error: %s", exc)
            return []

    raw_list = data.get("data") or []
    items: list[dict] = []
    for rank, entry in enumerate(raw_list, start=1):
        title = entry.get("title") or ""
        if not title:
            continue
        items.append({
            "title": title,
            "mal_id": entry.get("mal_id"),
            "jikan_rank": rank,
            "members": entry.get("members") or 0,
        })

    return items


def _fetch_reddit() -> list[dict]:
    """Fetch hot posts from r/anime and extract episode discussion threads."""
    url = "https://www.reddit.com/r/anime/hot.json?limit=50"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "anime-trend-bot/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            data = json.loads(resp.read().decode(charset, errors="replace"))
    except Exception:
        return []

    posts = (
        (data.get("data") or {})
        .get("children") or []
    )

    items: list[dict] = []
    for child in posts:
        post = (child.get("data") or {})
        post_title = post.get("title") or ""

        lower_title = post_title.lower()
        if "[episode]" not in lower_title and "[discussion]" not in lower_title:
            continue

        # Strip flair prefix like [Episode], [Discussion]
        clean = _STRIP_FLAIR_RE.sub("", post_title)
        # Strip episode number suffix like "- Episode 12"
        clean = _STRIP_EPISODE_RE.sub("", clean)
        # Strip trailing fluff characters
        clean = clean.strip(" -–—|·")

        if not clean:
            continue

        selftext = post.get("selftext") or ""
        selftext_snippet = selftext[:500]

        created_utc = post.get("created_utc") or 0
        posted_hours_ago = (time.time() - float(created_utc)) / 3600.0

        items.append({
            "title": clean,
            "post_title": post_title,
            "reddit_score": post.get("score") or 0,
            "num_comments": post.get("num_comments") or 0,
            "selftext_snippet": selftext_snippet,
            "posted_hours_ago": posted_hours_ago,
        })

    return items


# ---------------------------------------------------------------------------
# Scoring and merging
# ---------------------------------------------------------------------------

def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for fuzzy matching."""
    lowered = title.lower()
    stripped = _PUNCT_RE.sub("", lowered)
    return " ".join(stripped.split())


def _score_jikan_only(
    jikan_items: list[dict],
    reddit_items: list[dict],
    limit: int,
) -> list[dict]:
    """Fallback scorer when AniList is unavailable — uses Jikan as primary."""
    max_members = max((j["members"] for j in jikan_items), default=1) or 1
    reddit_by_title = {_normalize_title(r["title"]): r for r in reddit_items}

    scored = []
    for rank, item in enumerate(jikan_items, start=1):
        title = item["title"]
        norm = _normalize_title(title)
        jikan_score = item["members"] / max_members

        reddit_match = reddit_by_title.get(norm) or next(
            (v for k, v in reddit_by_title.items() if norm in k or k in norm), None
        )
        if reddit_match:
            reddit_score = min(math.log(1 + reddit_match["reddit_score"] + reddit_match["num_comments"]) / 10.0, 1.0)
        else:
            reddit_score = 0.0

        final_score = jikan_score * 0.7 + reddit_score * 0.3
        reason = f"#{rank} airing on MAL, {item['members']} members"
        if reddit_match:
            reason += f", {reddit_match['num_comments']} Reddit comments"

        scored.append({
            "title": title,
            "anilist_synopsis": "",
            "genres": [],
            "trending_reason": reason,
            "reddit_context": (reddit_match or {}).get("selftext_snippet", "")[:400],
            "score": round(min(1.0, final_score), 4),
            "urgency_score": 0.4,
            "sources": ["jikan"] + (["reddit"] if reddit_match else []),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def _score_candidates(
    anilist_items: list[dict],
    jikan_items: list[dict],
    reddit_items: list[dict],
    limit: int,
    cfg: dict | None = None,
) -> list[dict]:
    """Merge sources by normalized title and compute composite scores.

    Popularity is normalised on a log scale so niche seasonal shows
    cannot beat mainstream shows that happen to have a lower trending rank."""
    if not anilist_items and not jikan_items:
        return []

    if not anilist_items:
        return _score_jikan_only(jikan_items, reddit_items, limit)

    cfg = cfg or {}
    weights = cfg.get("score_weights") or {
        "anilist": 0.45, "jikan": 0.25, "reddit": 0.20, "popularity_norm": 0.10,
    }

    max_members = max((j["members"] for j in jikan_items), default=1) or 1
    max_popularity = max((a.get("popularity", 0) for a in anilist_items), default=1) or 1

    jikan_by_mal_id: dict[int, dict] = {
        j["mal_id"]: j for j in jikan_items if j.get("mal_id")
    }
    jikan_by_title: dict[str, dict] = {
        _normalize_title(j["title"]): j for j in jikan_items
    }

    reddit_by_title: dict[str, dict] = {}
    for r in reddit_items:
        key = _normalize_title(r["title"])
        if key not in reddit_by_title:
            reddit_by_title[key] = r

    episode_drop_bonus_hours = cfg.get("episode_drop_bonus_hours", 24)
    episode_drop_bonus_score = cfg.get("episode_drop_bonus_score", 0.15)

    scored: list[dict] = []
    for item in anilist_items:
        title = item["title"]
        norm = _normalize_title(title)
        rank = item["trending_rank"]

        anilist_score = max(0.0, 1.0 - (rank - 1) / 20.0)

        mal_id = item.get("mal_id")
        jikan_match = jikan_by_mal_id.get(mal_id) if mal_id else None
        if jikan_match is None:
            jikan_match = jikan_by_title.get(norm)
        if jikan_match is None:
            for jk, jv in jikan_by_title.items():
                if norm in jk or jk in norm:
                    jikan_match = jv
                    break
        jikan_score = (jikan_match["members"] / max_members) if jikan_match else 0.0

        reddit_match = reddit_by_title.get(norm)
        if reddit_match is None:
            for rk, rv in reddit_by_title.items():
                if norm in rk or rk in norm:
                    reddit_match = rv
                    break
        if reddit_match:
            rs = reddit_match["reddit_score"]
            rc = reddit_match["num_comments"]
            reddit_score = min(math.log(1 + rs + rc) / 10.0, 1.0)
        else:
            reddit_score = 0.0

        pop = item.get("popularity", 0)
        pop_score = math.log(1 + pop) / math.log(1 + max_popularity) if max_popularity > 0 else 0.0

        final_score = (
            anilist_score * weights["anilist"]
            + jikan_score * weights["jikan"]
            + reddit_score * weights["reddit"]
            + pop_score * weights.get("popularity_norm", 0.10)
        )

        edh = item.get("episode_drop_hours")
        if edh is not None and -6 <= edh <= 0:
            urgency_score = 1.0
        elif edh is not None and -episode_drop_bonus_hours <= edh <= 0:
            urgency_score = 0.7
        elif anilist_score > 0:
            urgency_score = 0.4
        else:
            urgency_score = 0.1

        if edh is not None and -episode_drop_bonus_hours <= edh <= 0:
            final_score = min(1.0, final_score + episode_drop_bonus_score)

        if edh is not None and edh <= 0:
            hours_ago = abs(round(edh, 1))
            reason = f"Episode aired {hours_ago}h ago, #{rank} on AniList"
            if reddit_match:
                reason += f", {reddit_match['num_comments']} Reddit comments"
        elif reddit_match:
            reason = f"#{rank} trending on AniList, {item['popularity']} members"
            reason += f", {reddit_match['num_comments']} Reddit comments"
        else:
            reason = f"#{rank} trending on AniList, {item['popularity']} members"

        reddit_context = ""
        if reddit_match:
            snippet = reddit_match.get("selftext_snippet") or ""
            reddit_context = snippet[:400]

        sources = ["anilist"]
        if jikan_match:
            sources.append("jikan")
        if reddit_match:
            sources.append("reddit")

        scored.append({
            "title": title,
            "anilist_synopsis": item["synopsis"],
            "genres": item["genres"],
            "trending_reason": reason,
            "reddit_context": reddit_context,
            "score": round(min(1.0, max(0.0, final_score)), 4),
            "urgency_score": urgency_score,
            "sources": sources,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]
