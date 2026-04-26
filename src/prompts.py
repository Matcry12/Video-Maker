"""Central prompt loader with template validation."""
import re
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache
def _load(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def render(name: str, **vars) -> str:
    tpl = _load(name)
    # Find all single-brace variables (not escaped {{...}})
    required = set(re.findall(r"(?<!\{)\{(\w+)\}(?!\})", tpl))
    missing = required - set(vars)
    if missing:
        raise ValueError(f"prompt '{name}' missing variables: {missing}")
    return tpl.format_map(vars)


def validate_all() -> None:
    """Assert every .txt file in prompts/ is non-empty. Called at process start."""
    for p in sorted(PROMPTS_DIR.glob("*.txt")):
        if p.stat().st_size == 0:
            raise ValueError(f"empty prompt file: {p}")
        _load(p.stem)
