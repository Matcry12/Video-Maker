# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Model Routing & Token Discipline

Default model is Opus 4.7, but **do not run every task on Opus**. Opus is expensive — reserve it for what actually needs it and delegate the rest to cheaper models, then review their output.

- **Stay on Opus (do directly):** architecture decisions, prompt/skill design, cross-pack reasoning, debugging subtle pipeline bugs, final review of delegated work.
- **Delegate to `sonnet`:** standard multi-file edits, refactors, test writing, focused implementation from a clear spec.
- **Delegate to `haiku`:** mechanical lookups, grep/find sweeps, file reads, boilerplate, single-command tasks.
- **Always review delegated output** before treating it as done — a subagent's summary describes intent, not result. Read the actual diff.

Spawn via the Agent tool with `model=haiku|sonnet`, or use `oh-my-claudecode:executor` (`model=opus` only for complex work). The goal: keep Opus context lean, push grunt work down, and verify on the way back up.

## Navigation Rules

**Off-limits during tasks (never read, modify, import, or create files here):**
- `archive/` — CV portfolio code only. If you need pipeline or web UI code, it does not exist for task purposes.
- `lab/` — experiments only. Never run or wire lab code unless the user explicitly says "apply to main flow".

**RAG over WebFetch (always):**
- Research a topic: `scripts/research_crawl.py` then `scripts/research_query.py`
- WebFetch eats 10K+ tokens/page and is never needed — the RAG cache handles it.

**Task → file map (use these, nothing else):**
| Task | Entry point |
|---|---|
| Render a Short | `scripts/render_script.py` |
| Render a Psychology-Facts stock-b-roll Short | `scripts/render_broll.py` (pkg `src/broll/`, skill `/broll`) |
| Render long-form video | `scripts/render_novel.py` |
| Render podcast (16:9 Remotion) | `scripts/render_podcast.py` |
| Re-render existing run | `scripts/render_from_run.py` |
| Research a topic | `scripts/research_crawl.py` → `scripts/research_query.py` |
| TTS synthesis | `src/tts.py` (Edge-TTS + Kokoro) or `src/tts_chatterbox.py` (voice cloning) |
| Long-form composition | `src/agent/long_editor.py` |
| Short composition | `src/editor.py` |
| YouTube metadata | `scripts/yt_metadata_skeleton.py` |
| Thumbnail | `src/thumbnail.py` |

**If a task fails mid-way:** search existing files before creating new ones. The file you need almost certainly exists — check `src/`, `src/agent/`, and `scripts/` first.

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

### Render pipeline (Shorts)

- TTS: `src/tts.py` — Edge-TTS with parallel chunking; PCM array concat (not FFmpeg copy) to avoid inter-chunk silence
- Subtitles: ASS format, generated in `src/editor.py`
- Video: PIL pre-compose → rawvideo pipe to FFmpeg (27x faster than N-overlay filter chain)
- BGM: mood-based selection from `assets/audio/bgm/`, mixed at 0.15 volume

### Long-form render pipeline (`/long-video` skill)

Two scripts handle long-form (10-15 min) videos:

**Full render** (TTS + subtitles + images + compose):
```
.venv/bin/python scripts/render_novel.py <script.json> <output_name>
```

**Re-render from existing wavs** (skips TTS, runs Whisper for subtitles):
```
.venv/bin/python scripts/render_from_run.py <run_dir> [output_name] [--mood <mood>]
```
- Both scripts auto-generate `<output_name>_thumb.jpg` and `youtube_metadata.txt`

**Long-form editor** (`src/agent/long_editor.py`):
- `compose_card()` — card-on-blurred-bg layout, contain-fit images (full image always visible, no cropping)
- Images are contain-fit: entire image scaled to fit card, blurred copy fills any padding — never cropped
- Karaoke word-level subtitles via ASS, chapter title overlays

**Post-render outputs** (auto-generated after every render):
- `output/videos/<name>.mp4` — final video
- `output/videos/<name>_thumb.jpg` — 1280×720 YouTube thumbnail (`src/thumbnail.py`)
- `output/runs/<run>/youtube_metadata.txt` — copy-paste ready title/description/chapters/hashtags/tags

**Script JSON shape** (`output/runs/<name>/script.json`):
- `chapters[]` — title, text, image_keywords, mood
- `youtube` — title, description, chapters, hashtags, tags

### Web UI

Flask app in `src/web.py`. Templates in `templates/` (Jinja2). Frontend JS in `static/js/`. Server passes data to JS via `APP_BOOTSTRAP` object. Jobs run in background threads; progress streamed via SSE.

## Conventions

- Before any refactor touching >3 files, write a `*_PLAN.md` at the project root and get approval before implementing.
- `AgentConfig` (user-facing overrides) flows through `_merge_config()` in `plan_agent.py` into `AgentPlan` (internal plan). `skill_id` lives on `AgentConfig`/`user_overrides`, NOT on `AgentPlan` — access it via `plan.user_overrides.skill_id`.
- Edit hooks may report false failures after successful tool results — trust the tool result, not the hook message.

## Testing Rules (MANDATORY)

- **Always test before reporting done.** After writing or changing code, run it and show actual output. Never claim something works without evidence.
  - Lab scripts: run `./.venv/bin/python lab/<path>/agent.py` and paste the output.
  - LLM prompts: call the model and paste the raw response.
  - Pipeline changes: trigger the relevant stage and verify the log line changes as expected.
- **Never commit untested code.** If a run fails, fix it first, then commit.
- **Lab before main.** All experiments go in `lab/`. Only move to `src/` when the user explicitly says "apply to main flow" after seeing test results.

## Explaining Problems (MANDATORY)

- **Always give a concrete example when describing a bug or problem.** Abstract descriptions are not enough.
  - Bad: "The image keywords are too generic."
  - Good: "Block text = 'Heavenly Restriction made Maki unable to use cursed energy'. Current output keywords: `['Heavenly', 'Restriction', 'She']`. Expected: `['Maki Zenin', 'Heavenly Restriction', 'Jujutsu Kaisen']`."
- Show before/after for every proposed fix so the user can judge whether it's worth doing.

