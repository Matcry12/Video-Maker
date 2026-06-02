"""Script + per-beat visual-keyword generation for the b-roll pipeline.

Three LLM calls:
1. write_script(topic, target_words)  -> tight narration split into beats,
   loaded from prompts/broll_writer.md.
2. visual_queries(beats, topic) -> per-beat concrete filmable search queries.
3. generate_metadata(topic, beats, hook) -> YouTube title/description/tags.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.llm_client import chat_completion
from src.agent_config import load_agent_settings

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of an LLM response."""
    if not text:
        raise ValueError("empty LLM response")
    text = text.strip()
    # strip code fences
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    # find first { or [
    m = re.search(r"[\[{]", text)
    if not m:
        raise ValueError(f"no JSON found in: {text[:200]}")
    snippet = text[m.start():]
    # try progressively shorter prefixes ending at matching bracket
    for end in range(len(snippet), 0, -1):
        try:
            return json.loads(snippet[:end])
        except Exception:
            continue
    raise ValueError(f"could not parse JSON from: {text[:200]}")


def _load_writer_brief() -> str:
    p = Path(__file__).resolve().parents[2] / "prompts" / "broll_writer.md"
    return p.read_text(encoding="utf-8")


def write_script(topic: str, target_words: int = 130) -> list[str]:
    """Write a punchy narration on `topic`, returned as a list of beats (sentences).

    Loads the writing brief from prompts/broll_writer.md at call time (lazy load).
    """
    system = "You write tight, punchy narration for faceless YouTube Shorts."
    user = _load_writer_brief().format(topic=topic, target_words=target_words)

    raw = chat_completion(
        system=system,
        user=user,
        stage="script",
        temperature=0.8,
        max_tokens=800,
    )
    data = _extract_json(raw)
    beats = [str(s).strip() for s in data if str(s).strip()]
    if not beats:
        raise ValueError("script generation returned no beats")
    return beats


def clickbait_hook(topic: str) -> str:
    """A short top-of-screen clickbait headline (3-6 words) for the video.

    Tuned for the psychology-facts niche: the headline opens a curiosity gap and,
    where natural, points at the viewer ("you / your brain") — the framing that
    drives saves & shares on this kind of content.
    """
    system = (
        "You write scroll-stopping top-screen hook headlines for psychology-facts "
        "YouTube Shorts. Your headlines open a curiosity gap the viewer NEEDS "
        "resolved."
    )
    user = (
        f'Topic: "{topic}"\n'
        "Write ONE top-screen hook headline. Rules:\n"
        "- 3 to 7 words, punchy, Title Case.\n"
        "- Open a curiosity gap: hint at a surprising answer without giving it.\n"
        "- Where natural, aim it at the viewer with \"You\" / \"Your Brain\" "
        "(makes them feel seen).\n"
        "- Prefer formats like \"Why You...\", \"The Real Reason You...\", "
        "\"3 Signs...\", \"What Your ... Says About You\".\n"
        "- No quotes, no emojis, no hashtags, no trailing period.\n"
        "Examples: \"Why You Wake Up At 3AM\", \"The Real Reason You Procrastinate\", "
        "\"3 Signs Someone Is Lying\".\n"
        "Return ONLY the headline text."
    )
    try:
        raw = chat_completion(
            system=system, user=user, stage="script",
            temperature=0.9, max_tokens=40,
        )
    except Exception as exc:
        logger.warning("clickbait_hook failed (%s); using topic", exc)
        return topic
    line = (raw or "").strip()
    line = line.splitlines()[0] if line else ""
    line = line.strip().strip('"').strip("'").rstrip(".").strip()
    return line[:60] or topic


def visual_queries(beats: list[str], topic: str) -> list[list[str]]:
    """For each beat, return up to `queries_per_beat` diverse stock-footage queries.

    Instead of a specificity gradient, the model is asked for DIVERSE CONCRETE
    ANGLES — different subjects, actions, and objects inside the same visual world.
    More diverse candidates gives the SigLIP reranker a richer pool to pick from.

    Two guards keep footage on-topic and prevent mismatches:
      1. The model first defines THIS topic's visual world (its natural footage
         palette). Every query must stay inside that world — flexible per topic
         (a wildlife topic *should* return animals), but no drift outside it.
      2. Queries depict the LITERAL meaning of each line. Figurative language is
         never filmed literally ("crave the hunt" -> human pursuit, not a hunting
         animal. "the rat race" -> commuters/crowds, not a rat).

    `queries_per_beat` is read from profiles/default.json agent.broll.queries_per_beat
    (default 5).
    """
    qpb: int = load_agent_settings().get("broll", {}).get("queries_per_beat", 5)
    numbered = "\n".join(f"{i}. {b}" for i, b in enumerate(beats))
    system = (
        f'You are a visual director choosing stock-footage search queries for a '
        f'short video about "{topic}".'
    )
    user = f"""STEP 1 — Define this video's VISUAL WORLD.
In one line, list the subjects, places and objects that naturally belong to THIS
topic's footage (its palette). Everything you search for must come from this world.

STEP 2 — For EACH numbered sentence, output up to {qpb} DIVERSE CONCRETE ANGLES as
Pexels search queries. Each angle should show a DIFFERENT subject, action, or object
inside the same visual world — not variations of the same shot.

Rules for every query:
- SHORT: 1 to 4 words, lowercase. Pexels matches tags, so long phrases return
  nothing. Write "man counting cash", NOT "stressed man counting cash at his desk".
- Concrete and filmable: real people, objects, places, actions you could point a
  camera at. No abstract concepts ("happiness", "success", "freedom").
- It MUST belong to the visual world from Step 1. Never introduce subjects from
  outside the topic. (e.g. no wild animals in a human/money video; no offices in
  a wildlife video; no people in a pure-space video.)
- Depict the sentence's LITERAL meaning. NEVER film a metaphor or figure of
  speech literally. "crave the hunt" -> the human pursuit it means, not a hunting
  animal. "the rat race" -> commuters/crowds, not a rat.
- Do not repeat the topic title verbatim.

Examples (sentence -> diverse angles):
- "Scientists call it the hedonic treadmill" ->
    ["man running treadmill", "person on treadmill", "gym workout", "scientist lab", "brain scan"]
- "True joy lives in the people you love" ->
    ["family laughing dinner", "friends hugging", "happy family", "couple holding hands", "children playing"]
- "Every paycheck just vanishes on bills" ->
    ["paying bills laptop", "counting money", "cash wallet", "credit card payment", "empty wallet"]

Sentences:
{numbered}

Return ONLY this JSON object:
{{"visual_world": "<one line>", "queries": [["q", ...], ...]}}
"queries" must have exactly {len(beats)} elements, in the same order as the sentences."""

    raw = chat_completion(
        system=system,
        user=user,
        stage="script",
        temperature=0.5,
        max_tokens=1200,
    )
    data = _extract_json(raw)

    # Accept either the {visual_world, queries} object or a bare array.
    if isinstance(data, dict):
        world = str(data.get("visual_world", "")).strip()
        rows = data.get("queries", [])
        if world:
            logger.info("visual world: %s", world)
    else:
        rows = data

    out: list[list[str]] = []
    for i in range(len(beats)):
        qs: list[str] = []
        if isinstance(rows, list) and i < len(rows) and isinstance(rows[i], list):
            seen: set[str] = set()
            for q in rows[i]:
                q = str(q).strip().lower()
                # keep queries short & Pexels-friendly: drop anything over 5 words
                # (a fixed cap — independent of how MANY queries we keep, qpb)
                if not q or q in seen or len(q.split()) > 5:
                    continue
                seen.add(q)
                qs.append(q)
        if not qs:
            # fallback: a couple of words from the beat itself
            words = re.findall(r"[a-zA-Z]{4,}", beats[i].lower())
            qs = [" ".join(words[:2])] if words else [topic]
        out.append(qs[:qpb])
    return out


def _normalize_hashtags(raw_tags: list, topic: str) -> list[str]:
    """Clean a hashtag list to the /writer convention: #Shorts first, deduped.

    Each entry is coerced to a single `#word` token (spaces stripped, one leading
    '#'). `#Shorts` is always forced to the front. Returns 8-10 tags.
    """
    out: list[str] = []
    seen: set[str] = set()
    for t in raw_tags:
        tok = re.sub(r"\s+", "", str(t)).lstrip("#")
        if not tok:
            continue
        tag = "#" + tok
        key = tag.lower()
        if key in seen or key == "#shorts":
            continue
        seen.add(key)
        out.append(tag)
    return ["#Shorts"] + out[:9]


def generate_metadata(topic: str, beats: list[str], hook: str) -> dict:
    """Return {title, description, hashtags, tags} for YouTube.

    Mirrors the /writer (qa) metadata shape: title <=70 chars curiosity-gap;
    description 2-3 short paragraphs + a follow CTA; `hashtags` a separate
    `#Shorts`-first block (8-10) that the uploader appends to the description;
    tags = 15-20 lowercase phrases (no '#').
    Falls back to a minimal dict on any failure.
    """
    script_preview = " ".join(beats[:4])
    system = "You write YouTube metadata for psychology-facts Shorts."
    user = (
        f'Topic: "{topic}"\n'
        f'Hook headline: "{hook}"\n'
        f'Script preview: "{script_preview}"\n\n'
        "Write YouTube metadata. Rules:\n"
        "- title: <=70 characters, curiosity-gap framing, no clickbait spam.\n"
        "- description: 2-3 short paragraphs explaining the video + one follow CTA "
        '  (e.g. "Follow for more psychology facts"). Do NOT put hashtags here.\n'
        "- hashtags: 7-9 relevant hashtags as a JSON list of '#word' tokens "
        '  (no spaces inside a tag). Do NOT include #Shorts (added automatically).\n'
        "- tags: list of 15-20 lowercase short phrases (no #), highly relevant.\n\n"
        "Return ONLY valid JSON:\n"
        '{"title": "...", "description": "...", "hashtags": ["#...", ...], "tags": ["...", ...]}'
    )
    try:
        raw = chat_completion(
            system=system, user=user, stage="script",
            temperature=0.7, max_tokens=600,
        )
        data = _extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError("expected dict")
        return {
            "title": str(data.get("title", hook))[:70],
            "description": str(data.get("description", topic)),
            "hashtags": _normalize_hashtags(data.get("hashtags", []), topic),
            "tags": [str(t) for t in data.get("tags", [])],
        }
    except Exception as exc:
        logger.warning("generate_metadata failed (%s); using fallback", exc)
        return {"title": hook, "description": topic, "hashtags": ["#Shorts"], "tags": []}
