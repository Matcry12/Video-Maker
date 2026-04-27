"""Centralized LLM client — Gemini primary, Groq fallback.

All LLM calls in the project should go through chat_completion() so that
provider failover is handled in one place.
"""

import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemma-3-27b-it"
_RATE_LIMIT_SLEEP = 2  # seconds between calls

# Circuit breaker state — module-level so it spans all callers in the process.
_consecutive_failures: dict[str, int] = {}  # provider -> count
_flipped_stages: set[str] = set()           # stages where provider order is flipped


@dataclass
class LLMResponse:
    """Response from chat_completion_with_meta — text + honest provenance."""
    text: str
    provider: str  # "gemini" | "groq"
    model: str


def chat_completion(
    *,
    system: str,
    user: str,
    stage: str | None = None,
    model: str = "",
    temperature: float = 0.3,
    timeout: float = 25.0,
    max_retries_429: int = 2,
    max_tokens: int = 2048,
) -> str:
    """Send a chat completion using per-stage provider + model config.

    Returns only the text. Use chat_completion_with_meta() if you also need
    to know which provider/model actually answered.
    """
    return chat_completion_with_meta(
        system=system,
        user=user,
        stage=stage,
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_retries_429=max_retries_429,
        max_tokens=max_tokens,
    ).text


def chat_completion_with_meta(
    *,
    system: str,
    user: str,
    stage: str | None = None,
    model: str = "",
    gemini_model: str = "",
    temperature: float = 0.3,
    timeout: float = 25.0,
    max_retries_429: int = 2,
    max_tokens: int = 2048,
) -> LLMResponse:
    """Chat completion that reports which provider/model answered.

    Provider order and model choice come from `stage` (resolved via
    profiles/default.json → models.stages). If `stage` is None, falls back
    to defaults. An explicit `model=` argument overrides the Groq model
    for this call only.

    Raises:
        RuntimeError: If all configured providers fail.
    """
    from .agent_config import resolve_stage

    cfg = resolve_stage(stage)
    groq_model = _resolve_groq_model(
        explicit=model,
        stage_default=cfg.groq_model,
        stage_provided=stage is not None,
    )
    gemini_model = gemini_model or cfg.gemini_model

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    providers = list(cfg.providers)
    if stage is not None and stage in _flipped_stages:
        providers = list(reversed(providers))

    no_think = bool(getattr(cfg, "no_think", False))

    last_exc: Exception | None = None
    for provider in providers:
        if provider == "gemini":
            if not gemini_key:
                continue
            try:
                text = _call_gemini(
                    system=system,
                    user=user,
                    api_key=gemini_key,
                    temperature=temperature,
                    model=gemini_model,
                    no_think=no_think,
                )
                _consecutive_failures["gemini"] = 0
                return LLMResponse(text=text, provider="gemini", model=gemini_model)
            except Exception as exc:
                last_exc = exc
                logger.warning("Gemini failed: %s. Trying next provider.", exc)
                _consecutive_failures["gemini"] = _consecutive_failures.get("gemini", 0) + 1
                if stage is not None and _consecutive_failures["gemini"] >= 3:
                    _flipped_stages.add(stage)
                    logger.warning(
                        "Circuit breaker: %s failed 3 times — flipping provider order for stage %r",
                        "gemini", stage,
                    )

        elif provider == "groq":
            if not groq_key:
                continue
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            try:
                text = _call_groq(
                    messages=messages,
                    model=groq_model,
                    api_key=groq_key,
                    temperature=temperature,
                    timeout=timeout,
                    max_retries_429=max_retries_429,
                    max_tokens=max_tokens,
                )
                _consecutive_failures["groq"] = 0
                return LLMResponse(text=text, provider="groq", model=groq_model)
            except _RateLimitError as exc:
                last_exc = exc
                logger.warning("Groq rate limited. Trying next provider.")
                _consecutive_failures["groq"] = _consecutive_failures.get("groq", 0) + 1
                if stage is not None and _consecutive_failures["groq"] >= 3:
                    _flipped_stages.add(stage)
                    logger.warning(
                        "Circuit breaker: %s failed 3 times — flipping provider order for stage %r",
                        "groq", stage,
                    )
            except Exception as exc:
                last_exc = exc
                logger.warning("Groq failed: %s. Trying next provider.", exc)
                _consecutive_failures["groq"] = _consecutive_failures.get("groq", 0) + 1
                if stage is not None and _consecutive_failures["groq"] >= 3:
                    _flipped_stages.add(stage)
                    logger.warning(
                        "Circuit breaker: %s failed 3 times — flipping provider order for stage %r",
                        "groq", stage,
                    )

        else:
            logger.warning("Unknown provider %r in stage %r — skipping.", provider, stage)

    raise RuntimeError(
        f"All providers failed for stage={stage!r} (tried {list(cfg.providers)}): {last_exc}"
    )


def _resolve_groq_model(*, explicit: str, stage_default: str, stage_provided: bool) -> str:
    """Resolve Groq model.

    When a stage is provided, the profile is authoritative —
    `explicit model=` > stage config. GROQ_MODEL env is ignored so the
    per-stage config isn't silently overridden.

    When no stage is provided (ad-hoc callers), precedence is
    `explicit model=` > GROQ_MODEL env > builtin default.
    """
    if explicit:
        return explicit
    if stage_provided:
        return stage_default
    return os.getenv("GROQ_MODEL", "").strip() or stage_default


def extraction_completion(
    *,
    system: str,
    user: str,
    temperature: float = 0.1,
    stage: str = "research_extract",
) -> str:
    """LLM call optimized for structured extraction.

    Defaults to stage="research_extract" (small/cheap Gemini model), with
    normal provider fallback. Pass a different stage to route elsewhere.
    """
    return chat_completion(
        system=system,
        user=user,
        stage=stage,
        temperature=temperature,
    )


class _RateLimitError(Exception):
    """Internal signal for Groq 429 rate limit."""
    pass


def _call_groq(
    *,
    messages: list[dict],
    model: str,
    api_key: str,
    temperature: float,
    timeout: float,
    max_retries_429: int,
    max_tokens: int = 2048,
) -> str:
    """Call Groq with retry on 429."""
    from groq import Groq

    client = Groq(api_key=api_key, timeout=timeout, max_retries=1)

    for attempt in range(max_retries_429 + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                stream=False,
                messages=messages,
                max_tokens=max_tokens,
            )
            choices = list(getattr(response, "choices", None) or [])
            if not choices:
                raise RuntimeError("Groq response has no choices.")
            content = getattr(choices[0].message, "content", "") or ""
            return content.strip()
        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "rate" in err_str:
                if attempt < max_retries_429:
                    wait = (attempt + 1) * 5
                    logger.warning("Groq 429, retry %d/%d in %ds...", attempt + 1, max_retries_429, wait)
                    time.sleep(wait)
                    continue
                raise _RateLimitError(str(exc))
            raise

    raise _RateLimitError("Groq rate limit exceeded after retries")


def _call_gemini(
    *,
    system: str,
    user: str,
    api_key: str,
    temperature: float,
    model: str = "",
    max_retries_503: int = 2,
    no_think: bool = False,
) -> str:
    """Call Gemini with configurable model (default: gemma-3-27b-it).

    Retries on 503 (service unavailable) with exponential backoff.
    Falls back to _GEMINI_MODEL if no model specified.

    When `no_think=True`, prepends `/no_think` to both the system and user
    content. Gemma 4 (31B) honours this prompt-level directive and skips the
    thinking channel — per Google's docs, the 31B sometimes still emits a
    thought channel if only one side asks, so we inject on both sides.
    """
    from google import genai
    from google.genai import types

    model_name = model or _GEMINI_MODEL
    client = genai.Client(api_key=api_key)

    if no_think:
        system = "/no_think\n" + system
        user = "/no_think\n\n" + user

    full_prompt = f"System instructions: {system}\n\nUser request:\n{user}"

    logger.info("Gemini: using %s", model_name)

    import re as _re

    last_exc: Exception | None = None
    max_attempts = max_retries_503 + 1
    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )

            text = response.text or ""
            if not text.strip():
                raise RuntimeError("Gemini returned empty response.")

            return text.strip()
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            is_503 = "503" in err_str or "unavailable" in err_str or "overloaded" in err_str
            is_429 = "429" in err_str or "resource_exhausted" in err_str or "rate_limit" in err_str
            if attempt < max_attempts - 1 and (is_503 or is_429):
                if is_429:
                    m = _re.search(r'retry in (\d+(?:\.\d+)?)s', str(exc), _re.IGNORECASE)
                    wait = min(float(m.group(1)) + 0.5, 60.0) if m else 6.0
                    logger.warning(
                        "Gemini 429, waiting %.1fs before retry %d/%d (model=%s)...",
                        wait, attempt + 1, max_retries_503, model_name,
                    )
                else:
                    wait = (attempt + 1) * 3
                    logger.warning(
                        "Gemini 503, retry %d/%d in %ds (model=%s)...",
                        attempt + 1, max_retries_503, wait, model_name,
                    )
                time.sleep(wait)
                continue
            raise

    raise last_exc
