"""Centralized LLM client — Gemini primary, Groq fallback.

All LLM calls in the project should go through chat_completion() so that
provider failover is handled in one place.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemma-3-27b-it"
_RATE_LIMIT_SLEEP = 2  # seconds between calls


def chat_completion(
    *,
    system: str,
    user: str,
    model: str = "",
    temperature: float = 0.3,
    timeout: float = 25.0,
    max_retries_429: int = 2,
) -> str:
    """Send a chat completion — tries Gemini first, falls back to Groq.

    Args:
        system: System prompt.
        user: User prompt.
        model: Groq model name (used only if falling back to Groq).
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        max_retries_429: How many times to retry Groq on 429 before giving up.

    Returns:
        The assistant's response text (stripped).

    Raises:
        RuntimeError: If both Gemini and Groq fail.
    """
    # --- Try Gemini first ---
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            return _call_gemini(
                system=system,
                user=user,
                api_key=gemini_key,
                temperature=temperature,
            )
        except Exception as exc:
            logger.warning("Gemini failed: %s. Falling back to Groq.", exc)

    # --- Fallback to Groq ---
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        raise RuntimeError(
            "Gemini failed and GROQ_API_KEY not set. "
            "Set GROQ_API_KEY to enable fallback."
        )

    if not model:
        model = os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        return _call_groq(
            messages=messages,
            model=model,
            api_key=groq_key,
            temperature=temperature,
            timeout=timeout,
            max_retries_429=max_retries_429,
        )
    except _RateLimitError:
        raise RuntimeError("Both Gemini and Groq failed (Groq rate limited).")
    except Exception as exc:
        raise RuntimeError(f"Both Gemini and Groq failed: {exc}")


_GEMINI_EXTRACT_MODEL = "gemma-3-4b-it"


def extraction_completion(
    *,
    system: str,
    user: str,
    temperature: float = 0.1,
) -> str:
    """LLM call optimized for structured extraction — uses Gemma 12B (cheaper, faster).

    Falls back to 27B if 12B is unavailable, then to Groq.
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            return _call_gemini(
                system=system,
                user=user,
                api_key=gemini_key,
                temperature=temperature,
                model=_GEMINI_EXTRACT_MODEL,
            )
        except Exception as exc:
            logger.warning("Gemini 12B extraction failed: %s. Trying 27B.", exc)
            try:
                return _call_gemini(
                    system=system,
                    user=user,
                    api_key=gemini_key,
                    temperature=temperature,
                    model=_GEMINI_MODEL,
                )
            except Exception as exc2:
                logger.warning("Gemini 27B also failed: %s. Falling back to Groq.", exc2)

    # Groq fallback
    return chat_completion(
        system=system,
        user=user,
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
) -> str:
    """Call Gemini with configurable model (default: gemma-3-27b-it).

    Retries on 503 (service unavailable) with exponential backoff.
    Falls back to _GEMINI_MODEL if no model specified.
    """
    from google import genai
    from google.genai import types

    model_name = model or _GEMINI_MODEL
    client = genai.Client(api_key=api_key)

    full_prompt = f"System instructions: {system}\n\nUser request:\n{user}"

    logger.info("Gemini: using %s", model_name)

    last_exc: Exception | None = None
    for attempt in range(max_retries_503 + 1):
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
            if "503" in err_str or "unavailable" in err_str or "overloaded" in err_str:
                if attempt < max_retries_503:
                    wait = (attempt + 1) * 3
                    logger.warning(
                        "Gemini 503, retry %d/%d in %ds (model=%s)...",
                        attempt + 1, max_retries_503, wait, model_name,
                    )
                    time.sleep(wait)
                    continue
            raise

    raise last_exc
