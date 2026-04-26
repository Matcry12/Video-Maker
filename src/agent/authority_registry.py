"""URL → authority tier classification.

Tier 1: canonical / primary sources.
Tier 2: editorial / vetted journalism.
Tier 3: user-generated / speculation.
Tier 4: unknown / uncategorized (default).
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Registry data
# ---------------------------------------------------------------------------

_TIER_1_DOMAINS = frozenset({
    # Global canonical
    "en.wikipedia.org", "wikipedia.org",
    # Anime / manga canonical
    "anilist.co", "myanimelist.net", "anime-planet.com", "kitsu.io",
    # Official publishers
    "crunchyroll.com", "viz.com", "shueisha.co.jp", "kadokawa.co.jp",
    # General knowledge canonical
    "britannica.com",
    # Government / education
    "nih.gov", "nasa.gov",
})

_TIER_2_DOMAINS = frozenset({
    # Entertainment editorial
    "screenrant.com", "cbr.com", "ign.com", "polygon.com", "kotaku.com",
    "animenewsnetwork.com", "crunchyroll.com/news",
    "gamespot.com", "pcgamer.com", "eurogamer.net", "rockpapershotgun.com",
    # News editorial
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com",
    "nytimes.com", "theguardian.com", "washingtonpost.com",
    "theverge.com", "arstechnica.com", "techcrunch.com",
})

_TIER_3_DOMAINS = frozenset({
    "reddit.com", "old.reddit.com", "medium.com", "substack.com",
    "quora.com", "tvtropes.org", "stackexchange.com", "stackoverflow.com",
    "forums.somethingawful.com",
})

# Heuristic patterns for tier 1 (topic-specific)
_FANDOM_PATTERN = re.compile(r"^[a-z0-9-]+\.fandom\.com$")


def _normalize_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower().strip()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def classify(url: str, topic_aliases: list[str] | None = None) -> int:
    """Return authority tier (1–4) for a URL.

    `topic_aliases` — list of topic name variants used to match
    topic-specific canonical subdomains (e.g. rezero.fandom.com
    for alias "rezero"). Pass an empty list if the topic is unknown.
    """
    domain = _normalize_domain(url)
    if not domain:
        return 4

    if domain in _TIER_1_DOMAINS:
        return 1

    # Subdomain walk for "foo.bar.tld" — match if "bar.tld" is tier 1 or 2
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        parent = ".".join(parts[i:])
        if parent in _TIER_1_DOMAINS:
            return 1
        if parent in _TIER_2_DOMAINS:
            return 2

    # Fandom wikis are tier 1 if the subdomain loosely matches a topic alias.
    # Shared-token check: ≥1 token of length ≥3 after stripping hyphens/underscores
    # matches between alias and subdomain. This handles "jujutsu-kaisen" ↔ "Jujutsu Kaisen"
    # and "JJK" ↔ "jujutsu" (no match, correctly stays tier 2).
    if _FANDOM_PATTERN.match(domain):
        if topic_aliases:
            sub = domain.split(".")[0]
            sub_tokens = set(re.findall(r"[a-z0-9]{3,}", sub.lower()))
            for alias in topic_aliases:
                alias_tokens = set(re.findall(r"[a-z0-9]{3,}", alias.lower()))
                if alias_tokens and sub_tokens and alias_tokens & sub_tokens:
                    return 1
        # Unmatched fandom wiki → tier 2 (not tier 1 — could be wrong franchise)
        return 2

    if domain in _TIER_2_DOMAINS:
        return 2
    if domain in _TIER_3_DOMAINS:
        return 3

    # Unrecognized user-generated indicators
    lowered = domain.lower()
    if any(tok in lowered for tok in ("blog.", "blogspot.", "wordpress.", "tumblr.")):
        return 3

    return 4


def tier_weight(tier: int, config: dict | None = None) -> float:
    """Score multiplier for a given tier. Reads weights from agent config
    if provided, else falls back to sensible defaults."""
    if config is None:
        config = {}
    weights = {
        1: config.get("tier_1_weight", 1.50),
        2: config.get("tier_2_weight", 1.00),
        3: config.get("tier_3_weight", 0.40),
        4: config.get("tier_4_weight", 0.25),
    }
    return weights.get(tier, 0.25)


def tier_label(tier: int) -> str:
    return {1: "canonical", 2: "editorial", 3: "user-generated", 4: "unknown"}.get(tier, "unknown")
