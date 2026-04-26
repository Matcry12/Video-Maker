# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Runtime

- Python: `./.venv/bin/python` — never bare `python` (not on PATH)
- Run the web UI: `./.venv/bin/python -m src.web` → http://localhost:5000
- Install deps: `./.venv/bin/pip install -r requirements.txt`
- Quality test: `./.venv/bin/python test_research_quality.py`
- API keys go in `.env` at project root (`GEMINI_API_KEY`, `GROQ_API_KEY`); loaded automatically by `src/web.py`

## Architecture

Fixed sequential pipeline — stages always run in this order:

```
plan → research → script → quality_gate → image → editor
```

Entry point: `src/agent/core.py` → `VideoAgent.run()`. Each stage is a function in `src/agent/*_agent.py`. Artifacts saved to `output/runs/<topic_timestamp>/` for debugging.

### LLM routing

All LLM calls go through `src/llm_client.py` → `chat_completion(stage=...)`. Stage routing lives in `profiles/default.json` under `"models"`. Never hardcode model names — pass `stage=` and let `resolve_stage()` pick.

```python
from src.agent_config import resolve_stage      # → StageModelCfg(providers, gemini_model, groq_model)
from src.agent_config import load_agent_settings  # → dict of runtime tuning knobs
```

Stage names and their models (from `profiles/default.json`):

| Stage | Provider | Model |
|---|---|---|
| `script`, `quality_gate`, `refine`, `research` | Groq → Gemini | llama-3.3-70b / gemma-3-27b-it |
| `research_extract` | Groq → Gemini | llama-3.3-70b / gemma-3-27b-it |
| `plan`, `crawl`, `interest_rank`, `bank_extract`, `research_eval` | Groq only | llama-3.1-8b-instant |

Provider order: Groq primary → Gemini fallback (per stage). Precedence: explicit `model=` kwarg > stage config > builtin default.

### Profile config (`profiles/default.json`)

Three sections:
- `"models"` — per-stage LLM routing (`providers`, `groq_model`, `gemini_model`)
- `"agent"` — runtime tuning knobs (`images_per_block`, `tts`, `editor`, `research` depth settings)
- Everything else — video output (resolution, fps, TTS defaults, subtitle presets)

Add new tunables under `"agent"` and read via `load_agent_settings().get("key", default)`. Never hardcode magic numbers in source files.

### Skill templates (`skills/*.json`)

13 templates define video style (hook rules, pacing, tone, prompt injection, extraction hint). Selected automatically via BM25 in `src/agent/skill_selector.py`, or forced via `AgentConfig.skill_id`.

Each skill has an `extraction_hint` field — a 1-2 sentence guide used by the research extraction prompt to prioritize relevant facts. `skill_id` flows: `AgentConfig.skill_id` → `plan.user_overrides.skill_id` → `run_research(skill_id=...)` → `_load_extraction_hint(skill_id)`.

To add a new skill, add `skills/<name>.json` with all required fields including `extraction_hint`.

### Research pipeline (`src/agent/research_agent.py`)

3-stage pipeline: DuckDuckGo + Wikipedia → Crawl4AI (BM25 scored) → per-page LLM extraction → Jaccard dedup → reflect/crawl loop.

Key details:
- Extraction prompt lives in `prompts/research_extract.txt`, loaded at runtime via `_load_prompt()`
- `_PAGE_TEXT_MAX_CHARS` (default 3000 from profile) controls how much of each page is sent to the LLM
- Low-quality domains (merch stores, shops) are filtered before extraction via `_is_low_quality_url()`
- `_stage_dedup_and_format()` uses word-level Jaccard similarity (threshold 0.5) to drop near-duplicate facts
- `run_research()` signature: `(topic, search_queries, language, skill_id="", emit=None)`

### Render pipeline

- TTS: `src/tts.py` — Edge-TTS with parallel chunking; PCM array concat (not FFmpeg copy) to avoid inter-chunk silence
- Subtitles: ASS format, generated in `src/editor.py`
- Video: PIL pre-compose → rawvideo pipe to FFmpeg (27x faster than N-overlay filter chain)
- BGM: mood-based selection from `assets/audio/bgm/`, mixed at 0.15 volume

### Web UI

Flask app in `src/web.py`. Templates in `templates/` (Jinja2). Frontend JS in `static/js/`. Server passes data to JS via `APP_BOOTSTRAP` object. Jobs run in background threads; progress streamed via SSE.

## Conventions

- Before any refactor touching >3 files, write a `*_PLAN.md` at the project root and get approval before implementing.
- `AgentConfig` (user-facing overrides) flows through `_merge_config()` in `plan_agent.py` into `AgentPlan` (internal plan). `skill_id` lives on `AgentConfig`/`user_overrides`, NOT on `AgentPlan` — access it via `plan.user_overrides.skill_id`.
- Edit hooks may report false failures after successful tool results — trust the tool result, not the hook message.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
