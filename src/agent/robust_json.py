"""Safe JSON extraction from LLM responses.

LLMs frequently wrap JSON in prose, markdown fences, or trailing explanations.
`re.search(r'\\{.*\\}', resp, re.DOTALL)` fails when multiple objects appear
or when there's trailing text inside the last brace. Use this helper instead.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_fences(raw: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences, keep inner content."""
    m = _FENCE_RE.search(raw)
    return m.group(1).strip() if m else raw.strip()


def extract_first_json(raw: str) -> dict | list | None:
    """Return the first valid JSON object OR array found in `raw`, or None.

    Strategy:
      1. Strip markdown code fences.
      2. Try full `json.loads(stripped)`.
      3. Walk the string looking for '{' or '['; for each, try
         `raw_decode()` starting there. Return first success.
    """
    if not raw or not isinstance(raw, str):
        return None

    cleaned = _strip_fences(raw)

    # Fast path: whole string is valid JSON
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass

    # Slow path: find first '{' or '[' and try raw_decode at each
    decoder = json.JSONDecoder()
    for start in range(len(cleaned)):
        ch = cleaned[start]
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(obj, (dict, list)):
                return obj
        except json.JSONDecodeError:
            continue

    return None


def extract_json_dict(raw: str, required_keys: list[str] | None = None) -> dict[str, Any] | None:
    """Same as extract_first_json but only returns dicts; validates required keys.

    Returns None (and logs at DEBUG) if the result is not a dict or is missing
    any required key. Callers should log their own WARNING if they care.
    """
    obj = extract_first_json(raw)
    if not isinstance(obj, dict):
        logger.debug("extract_json_dict: no dict found in %d chars", len(raw or ""))
        return None
    if required_keys:
        missing = [k for k in required_keys if k not in obj]
        if missing:
            logger.debug("extract_json_dict: missing keys %s", missing)
            return None
    return obj


def extract_json_list(raw: str) -> list | None:
    obj = extract_first_json(raw)
    return obj if isinstance(obj, list) else None


def parse_yes_no(value: Any) -> bool:
    """True iff value starts with 'yes' (case-insensitive, whitespace/punct tolerant).

    Handles: 'yes', 'Yes', 'YES', 'yes.', 'yes!', '  Yes  ', 'yes, because ...'
    """
    if value is True:
        return True
    if not isinstance(value, str):
        return False
    stripped = value.strip().lower()
    return bool(re.match(r"^yes\b", stripped))
