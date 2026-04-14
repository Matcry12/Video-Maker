# Video Maker - AI Video Generation Pipeline

An end-to-end pipeline that transforms a text prompt into a fully rendered YouTube Shorts video with narration, subtitles, relevant images, and background music.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Pipeline Flow](#pipeline-flow)
4. [Pipeline Stages](#pipeline-stages)
5. [Core Algorithms](#core-algorithms)
6. [Tech Stack](#tech-stack)
7. [Configuration](#configuration)
8. [Web Interface](#web-interface)
9. [Performance Optimizations](#performance-optimizations)
10. [Project Structure](#project-structure)
11. [Limitations & Honest Assessment](#limitations--honest-assessment)

---

## Overview

**What it does:** Takes a text prompt like *"Dark secrets about Takaba in Jujutsu Kaisen"* and:
- Researches the topic across the web (DuckDuckGo, Wikipedia, Crawl4AI)
- Extracts and ranks facts by viral potential using LLM reranking
- Writes a narration script optimized for short-form video
- Generates speech with word-level timestamps
- Finds and orders relevant images matched to the narration timeline
- Renders the final video with subtitles, Ken Burns effects, and background music

**Target format:** Vertical video (1080x1920), 30fps, optimized for YouTube Shorts / TikTok / Instagram Reels.

**Architecture style:** Fixed sequential pipeline with LLM calls at each stage. Not a true agent system — the stages always run in the same order (plan -> research -> script -> quality check -> images -> render). Each stage has fallback chains and retry loops for robustness.

---

## System Architecture

```
                         User Prompt
                              |
                    +---------v----------+
                    |   PLAN STAGE       |
                    | (topic extraction, |
                    |  language, style)  |
                    +---------+----------+
                              |
                    +---------v----------+
                    |  RESEARCH STAGE    |
                    | (multi-source RAG) |
                    |  DDG + Wiki +      |
                    |  Crawl4AI + BM25   |
                    |  + LLM extraction  |
                    +---------+----------+
                              |
                    +---------v----------+
                    |   SCRIPT STAGE     |
                    | (skill selection,  |
                    |  fact-to-narration, |
                    |  lint + rewrite)   |
                    +---------+----------+
                              |
                    +---------v----------+
                    |   QUALITY GATE     |
                    | (3-question eval,  |
                    |  retry on failure) |
                    +---------+----------+
                              |
                    +---------v----------+
                    |   IMAGE STAGE      |
                    | (per-block search, |
                    |  download, match)  |
                    +---------+----------+
                              |
                    +---------v----------+
                    |   RENDER STAGE     |
                    |  TTS + Subtitles   |
                    |  + Video Compose   |
                    |  + BGM Mixing      |
                    +---------+----------+
                              |
                         Final Video
                     (.mp4 + .mp3 + artifacts)
```

**Note:** The code internally uses the naming convention `*_agent.py` for each stage (e.g., `plan_agent.py`, `research_agent.py`). This is a naming convention from early development — these modules are pipeline stages, not autonomous agents. They don't make decisions about what to do next or dynamically choose tools. Each stage runs a fixed sequence of operations and passes its output to the next stage.

---

## Pipeline Flow

### Phase 1: Planning

**File:** `src/agent/plan_agent.py`

The planning stage parses the user's free-form prompt and extracts structured metadata:

| Field | Example | Purpose |
|-------|---------|---------|
| `topic` | "Takaba Jujutsu Kaisen" | Core subject for research |
| `language` | "vi-VN" | Target language for script + TTS |
| `content_type` | "dark_secrets" | Used for skill template matching |
| `mood` | "dark_mystery" | Determines BGM selection |
| `hook_strategy` | "shock" | Script opening style |
| `search_queries` | ["Takaba JJK hidden power", ...] | 6-8 diverse search angles |

**How it works:**
1. Sends prompt to a lightweight LLM (Groq 8b-instant) for structured JSON extraction
2. Applies heuristic fallbacks (regex for language detection, keyword matching for mood)
3. Merges any explicit user config overrides (voice, image mode, style)

The planning stage generates diverse search queries covering different angles: dark facts, hidden details, fan theories, character analysis, and creator interviews.

---

### Phase 2: Research

**File:** `src/agent/research_agent.py`

The research stage runs a 3-stage pipeline to gather high-quality, topic-relevant facts:

#### Stage 1: Search + Crawl

```
DDG Search (12 queries) ──> URLs + Snippets
                                  |
Wikipedia Fetch ─────────────>   Combined
(target lang + English)           Source Pool
                                  |
Crawl4AI (top URLs) ────────>   BM25-filtered
  with BM25ContentFilter          Markdown
```

- **DuckDuckGo:** 12 diverse queries generate up to 60 unique source URLs
- **Wikipedia:** Full article fetch in both target language and English, combined into one extraction unit
- **Crawl4AI:** Async web crawling with `BM25ContentFilter` — the crawler itself filters content by topic relevance during extraction, producing focused `fit_markdown` instead of full noisy pages
- **URL filtering:** Skips Pinterest, TikTok, Instagram, YouTube, and other social media (low-quality for fact extraction)

#### Stage 2: LLM Extraction

```
Each crawled page ──> compress_for_llm() ──> Gemma 4B extraction
                      (cap 1500 chars)        (anti-hallucination prompt)
                                                    |
                                              Topic-specific facts
                                              with source attribution
```

- Each page is independently processed by **Gemma 3 4B** (cheap, fast, good at extraction)
- Anti-hallucination prompt: *"Extract ONLY facts explicitly stated in the text. Do NOT invent, infer, or combine information."*
- `compress_for_llm()` pre-cleaning removes boilerplate, ads, navigation before sending to the model
- Capped at 6 pages total (`_MAX_EXTRACT_PAGES`) to control cost
- Model fallback chain: Gemma 4B -> Gemma 27B -> Groq

#### Stage 3: Dedup + Format

- Removes duplicate facts across sources
- Formats into structured fact objects with source URLs
- Tags each fact with relevance score and suggested role (hook, setup, reveal, etc.)

**Why this 3-stage approach?**

The original approach used RAG chunking (split pages into 500-char chunks, index with BM25, retrieve top-K). This failed because chunking destroys entity context — a chunk about "Hakari's domain" from a JJK wiki page would rank higher than the actual target character's facts, causing fact contamination. The LLM extraction stage reads the full page context and extracts only topic-specific facts.

---

### Phase 3: Script Writing

**File:** `src/agent/script_agent.py`, `src/content_sources/fact_script_writer.py`

#### Skill Selection

Before writing, the system matches the plan metadata to a video type template from `skills/*.json`:

| Skill | Video Type | Key Differentiator |
|-------|-----------|---------------------|
| `_default` | General viral narration | Fast-paced, shock hook first |
| `did_you_know` | "Did you know?" | Curiosity triggers, revelation pacing |
| `dark_secrets` | Dark secrets / hidden truths | Dark tone, escalating shock |
| `top_list` | Top N countdown | Numbered countdown, "#1 reveal" |
| `theory` | Fan theory analysis | Debate framing, evidence presentation |
| `comparison` | X vs Y battle | Side-by-side, "winner" reveal |
| `lore_deep_dive` | Deep lore exploration | Worldbuilding, connected details |

Each skill defines:
- **Hook rule:** How to open the video
- **Pacing rule:** Rhythm between facts
- **Ending rule:** How to close and create a loop
- **Transitions:** Language-specific connecting phrases
- **Prompt injection:** Raw instruction appended to the LLM prompt

Selection uses BM25 matching on skill descriptions + trigger keywords, with bonuses for matching `content_type` and `mood` from the plan.

#### Script Generation

The fact-to-script writer (`fact_script_writer.py`) sends facts + skill template to Groq LLM (llama-3.3-70b-versatile) and generates narration blocks with roles:

```
hook     -> "There's a terrifying secret about Takaba..."
setup    -> "His technique 'Comedian' is ranked alongside Gojo's Limitless..."
twist    -> "But Hazenoki, a powerful sorcerer, surrendered just because of laughter..."
reveal   -> "The technique doesn't just create laughter. It's based on absolute confidence..."
payoff   -> "And the scariest part? He might not actually be dead..."
```

Each block includes `image_keywords` — a chronologically ordered list of what the viewer should see at each moment.

#### Script Linting

`script_lint.py` runs deterministic quality checks:
- Readability score (sentence length distribution)
- Total length within target range
- Pacing validation (no blocks too long or too short)
- Language consistency (script matches target language)
- Repetition detection (no duplicate phrases)

If linting fails, the script is rewritten with specific lint feedback injected into the prompt. This is one of the few feedback loops in the pipeline — the lint result directly influences the next LLM call.

---

### Phase 4: Quality Gate

**File:** `src/agent/quality_gate.py`

A 3-question LLM evaluation:
1. *"Is this suitable for viral YouTube Shorts?"*
2. *"Is the pacing engaging (not too slow)?"*
3. *"Is the script in the correct target language?"*

Requires 2/3 "yes" to pass. On failure, the script stage reruns with quality feedback. This is the second feedback loop — quality gate failure triggers a script rewrite.

---

### Phase 5: Image Retrieval

**File:** `src/agent/image_agent.py`, `src/images/pipeline.py`

#### Search Strategy

```
Per block:
  1. Topic-first search (just topic name) ──> base image pool
  2. Per-keyword searches (block.image_keywords) ──> keyword-matched images
  3. Fallback: DDG-only retry if multi-source fails
  4. Fallback: Remove deduplication (accept reuse over gaps)
```

Image sources: DuckDuckGo Images, Pixabay, Wikimedia Commons (selected by topic category).

#### Image-to-Timeline Matching

Images are sorted to match the narration timeline using `image_keywords` chronological order:

```python
# Each image carries its search keyword
{"path": "tmp/cache/images/abc123.jpg", "keyword": "Takaba JJK manga panel"}

# Keywords are chronological (verified from LLM output):
# keyword[0] = "Takaba JJK manga panel"         -> narration start
# keyword[1] = "Iori Hazenoki JJK defeated"     -> middle
# keyword[2] = "Takaba JJK corpse funeral scene" -> narration end

# Images sorted by keyword index -> appear in narration order
```

**Why this matters:** Without sorting, images appear in download order. The narrator says "Hazenoki surrendered because of laughter" but the image shown might be a random portrait from the beginning. With sorting, the image matches the current narration moment.

The matching algorithm:
1. For each image with a keyword field: direct match to keyword index
2. For images without keyword: BM25/substring match against segment text
3. Unmatched images fill remaining gaps
4. Processing time: <10ms for 30 images (no ML model, just string matching)

---

### Phase 6: Video Rendering

**File:** `src/agent/editor_agent.py`, `src/manager.py`, `src/editor.py`

#### TTS (Text-to-Speech)

**File:** `src/tts.py`

Uses Microsoft Edge-TTS (free, high quality, supports Vietnamese and 40+ languages).

**Parallel chunking** for long texts:
1. Split text into sentence-level chunks (~400 chars each)
2. Synthesize chunks concurrently (up to 5 parallel edge-tts calls)
3. Merge audio via PCM concatenation (numpy array concat — gapless, no MP3 frame alignment issues)
4. Merge word-level timestamps with offset correction

**Alignment modes:**
- **corrected** (default): Post-processes edge-tts WordBoundary timestamps with timing smoothing and drift correction. Ground truth from the TTS engine itself — 0.000s drift.
- **edge**: Raw edge-tts timestamps without post-processing.
- **forced**: Uses faster-whisper to re-transcribe the audio and align words via fuzzy matching. Slower and less accurate for non-English (43s drift observed for Vietnamese).

#### Subtitle Generation

**File:** `src/editor.py` — `generate_ass()`

Generates ASS (Advanced SubStation Alpha) subtitle files with:
- **Phrase-level segmentation:** Words grouped into captions respecting `max_chars_per_line`, `max_duration`, and `max_cps` (characters per second)
- **Keyword highlighting:** Configurable highlight words rendered with a separate ASS style
- **Line rebalancing:** Avoids orphaned short words at line starts
- **Timing guardrails:** Minimum event duration (0.35s), maximum gap between captions (0.24s)

Subtitle presets (from profile):
| Preset | Max Chars | Duration Range | CPS |
|--------|-----------|----------------|-----|
| minimal | 30 | 1.0 - 2.7s | 15 |
| energetic | 24 | 0.8 - 2.1s | 17 |
| cinematic | 26 | 1.2 - 3.0s | 14 |

#### Video Composition

Two image display modes:

**Popup mode** (default for storytelling):
- Background video loop with floating image overlays
- Pop sound effect at image appearance
- Images appear as styled popups over the background

**Background mode** (for scenic/visual topics):
- Full-screen images with Ken Burns pan effect (6 pan directions)
- Images fill the entire frame with subtle motion
- Hard cuts between images (no alpha fade)

**Rendering pipeline:**
1. Per-block rendering: TTS audio + subtitles + images + background -> individual `.mp4` clips
2. Concat phase: FFmpeg concat demuxer chains all block clips
3. BGM mixing: Mood-specific background music from `assets/audio/bgm/{mood}/` mixed at configured volume
4. Audio export: Separate `.mp3` track

**Hardware acceleration:** NVENC (NVIDIA) when available, CPU fallback otherwise.

---

## Core Algorithms

### BM25 Sparse Retrieval

**File:** `src/content_sources/rag_index.py`

Used in two places:
1. **Content crawling:** `BM25ContentFilter` in Crawl4AI filters web page content by topic relevance during extraction
2. **Skill matching:** Indexes skill descriptions + trigger keywords for template selection

```python
# Tokenization: lowercase, filter words > 1 char
# Scoring: BM25Okapi (k1=1.5, b=0.75)
# Retrieval: top-K ranked by BM25 score
```

### Hybrid Retrieval (BM25 + Dense)

**File:** `src/content_sources/rag_index.py`

Optional dense embedding layer using fastembed (ONNX-based, CPU):
- Model: `BAAI/bge-small-en-v1.5` (~130MB)
- Combination: Reciprocal Rank Fusion (RRF) merges BM25 and dense rankings
- Score formula: `RRF(doc) = 1/(k + rank_bm25) + 1/(k + rank_dense)` where k=60

### Topic Entity Extraction

Identifies which facts actually mention the target entity:
- Splits topic into searchable name variants ("Takaba", "JJK Takaba", etc.)
- Penalizes facts that don't mention any variant (penalty = 0.1x score)
- Prevents fact contamination from tangentially related content

### PCM Audio Concatenation

For gapless TTS chunk merging:
```python
# Decode each MP3 chunk to PCM (numpy array)
# Concatenate arrays directly (no re-encoding gaps)
# Write merged PCM to final output
# Adjust word timestamps by cumulative chunk duration offset
```

This avoids the inter-chunk silence that FFmpeg `-c copy` on MP3 produces due to MP3 frame alignment.

---

## Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.13+ | Core runtime |
| **LLM** | Google Gemini (Gemma 4B/27B) | Fact extraction, script writing |
| **LLM Fallback** | Groq (Llama 3.3 70B) | Script writing, reranking |
| **TTS** | Microsoft Edge-TTS | Speech synthesis (40+ languages) |
| **Alignment** | faster-whisper | Optional forced alignment |
| **Video** | FFmpeg | Rendering, composition, encoding |
| **Web Crawling** | Crawl4AI | Async headless browser crawling |
| **Search** | DuckDuckGo (ddgs) | Web + image search |
| **RAG** | rank-bm25, fastembed | Sparse + dense retrieval |
| **Image Processing** | Pillow/PIL | Resize, format conversion, caching |
| **Web UI** | Flask + Jinja2 | Backend + templating |
| **Validation** | Pydantic | Type-safe config and models |
| **Audio** | soundfile, numpy | Audio processing and concatenation |

### LLM Model Routing

| Task | Model | Why |
|------|-------|-----|
| Plan extraction | Groq 8b-instant | Cheap, fast, structured output |
| Fact extraction | Gemma 3 4B | Cheap, zero rate limits, good at extraction |
| Script writing | Groq Llama 3.3 70B | High quality narration |
| Quality gate | Groq 8b-instant | Simple yes/no evaluation |
| Reranking | Groq 8b-instant | Score comparison |

Fallback chain: Gemma 4B -> Gemma 27B -> Groq (automatic failover on 503/rate limits).

---

## Configuration

### Profile System

**File:** `profiles/default.json`

Profiles control video output parameters:

```json
{
  "resolution": {"width": 1080, "height": 1920},
  "fps": 30,
  "default_voice": "NamMinh",
  "default_background": "assets/videos/",
  "tts": {
    "default_rate": "+25%",
    "default_alignment_mode": "corrected"
  },
  "subtitle": {
    "default_preset": "minimal"
  },
  "global_audio": {
    "bgm_folder": "assets/audio/bgm/",
    "default_bgm_volume": 0.15
  }
}
```

### BGM Mood System

Background music is auto-selected based on the script's `bgm_mood` field:

```
assets/audio/bgm/
  intense/    -> action, battle topics
  calm/       -> educational, informational
  mystery/    -> dark secrets, hidden truths
  epic/       -> power rankings, legendary events
  emotional/  -> character deaths, tragic stories
```

A random track from the matching mood folder is selected for each video.

---

## Web Interface

**File:** `src/web.py`, `templates/index.html`

Three-tab interface:

### Generate Tab (Autonomous Mode)
- Free-form text prompt input
- Real-time progress tracking: Plan -> Research -> Script -> Quality -> Image -> Render
- AJAX polling on job status
- Video preview and download on completion

### Editor Tab (Manual Mode)
- Block-by-block script editing
- Subtitle timing preview
- Manual image assignment
- Direct render trigger

### Crawl Tab (Research Mode)
- Manual fact entry
- Source draft composition
- Content bank management

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/generate` | POST | Start pipeline |
| `/api/jobs/{id}` | GET | Job status + progress |
| `/api/scripts/{name}` | GET/POST | Load/save scripts |
| `/output/{file}` | GET | Download final video/audio |

---

## Performance Optimizations

| Optimization | Before | After | Improvement |
|-------------|--------|-------|-------------|
| Video rendering (PIL pre-compose) | 407s | 14.8s | 27.5x faster |
| TTS (parallel chunking) | baseline | 2.18x | Parallel sentence synthesis |
| Subtitle drift (corrected alignment) | 43s drift | 0.000s | Ground-truth timestamps |
| Research relevance (LLM extraction) | ~43% | 67-100% | Full-page context extraction |

### Caching Strategy

- **TTS cache:** SHA256 hash of (text + voice + rate + pitch) -> cached MP3 + word timestamps
- **Image cache:** Downloaded images stored in `tmp/cache/images/` by content hash
- **Resize cache:** PIL-downscaled images in `tmp/cache/resized/` by (filename + target width)
- **BM25 index:** Ephemeral in-memory (built per query, cleared after retrieval)

### Memory Management

- `gc.collect()` after each block render (release overlay data)
- Concurrent TTS limited by semaphore (5 local, 10 global)
- LLM extraction capped at 6 pages per research run
- fastembed model cached at module level (loaded once)

---

## Project Structure

```
Video maker/
|
+-- profiles/
|   +-- default.json              # Video output configuration
|
+-- skills/                       # Script writing templates
|   +-- _default.json             # General viral narration
|   +-- did_you_know.json         # "Did you know?" format
|   +-- dark_secrets.json         # Dark secrets format
|   +-- top_list.json             # Top N countdown
|   +-- theory.json               # Fan theory analysis
|   +-- comparison.json           # X vs Y battle
|   +-- lore_deep_dive.json       # Deep lore exploration
|   +-- easter_eggs.json          # Hidden references
|
+-- src/
|   +-- __init__.py               # .env loader
|   +-- llm_client.py             # Gemini/Groq with fallback chains
|   +-- manager.py                # VideoManager (orchestration + render)
|   +-- editor.py                 # FFmpeg render, ASS subtitles, Ken Burns
|   +-- tts.py                    # Edge-TTS, parallel chunking, alignment
|   +-- whisper_align.py          # Forced alignment via faster-whisper
|   +-- web.py                    # Flask web application
|   |
|   +-- agent/                    # Pipeline stages (named "agent" by convention)
|   |   +-- models.py             # Pydantic models (AgentPlan, AgentState, etc.)
|   |   +-- core.py               # Pipeline orchestrator (6-phase sequence)
|   |   +-- plan_agent.py         # Prompt -> structured plan
|   |   +-- research_agent.py     # Multi-source RAG research
|   |   +-- script_agent.py       # Facts -> narration script
|   |   +-- quality_gate.py       # 3-question quality evaluation
|   |   +-- image_agent.py        # Image search + matching
|   |   +-- editor_agent.py       # Script -> rendered video
|   |   +-- skill_selector.py     # BM25-based skill template matching
|   |
|   +-- content_sources/
|   |   +-- crawl4ai_source.py    # Async web crawling with BM25 filter
|   |   +-- rag_index.py          # BM25 + dense embedding indices
|   |   +-- fact_script_writer.py # Facts -> script blocks via LLM
|   |   +-- script_lint.py        # Deterministic script quality checks
|   |   +-- duckduckgo_source.py  # DuckDuckGo search integration
|   |   +-- wikipedia_source.py   # Wikipedia article fetching
|   |
|   +-- images/
|       +-- pipeline.py           # Per-block image search + download
|       +-- fetch.py              # Multi-source image search (DDG/Pixabay/Wiki)
|
+-- templates/                    # Jinja2 HTML templates
|   +-- index.html                # Main UI (3-tab interface)
|   +-- partials/                 # Reusable template components
|
+-- static/                       # Frontend assets
|   +-- css/                      # Stylesheets
|   +-- js/                       # JavaScript modules
|
+-- assets/
|   +-- audio/bgm/               # Background music by mood
|   +-- audio/sound_effects/     # Pop sound for image popups
|   +-- videos/                   # Default background videos
|
+-- output/                       # Generated videos
|   +-- runs/{run_id}/           # Per-run artifacts (plan, research, script JSONs)
|
+-- tmp/cache/                    # Runtime caches (TTS, images, resized)
```

---

## How to Run

### Prerequisites
- Python 3.13+
- FFmpeg installed and in PATH
- API keys set in `.env`:
  - `GEMINI_API_KEY` (Google AI Studio)
  - `GROQ_API_KEY` (Groq Cloud)

### Installation
```bash
pip install -r requirements.txt
```

### Usage

**Web UI (recommended):**
```bash
python -m src.web
# Open http://localhost:5000
```

**Programmatic:**
```python
from src.agent.core import VideoAgent

pipeline = VideoAgent()
result = pipeline.run("Dark secrets about Takaba in Jujutsu Kaisen, Vietnamese")
print(result.video_path)  # output/takaba_dark_secrets.mp4
```

---

## Key Design Decisions

| Decision | Alternative Considered | Why Chosen |
|----------|----------------------|------------|
| Gemma 4B for extraction | Gemma 12B/27B | 4B has zero rate limits on free tier, 12B gets 503s constantly |
| BM25 over CLIP for image matching | CLIP/SigLIP embeddings | CLIP requires 2-4GB VRAM, BM25 is <10ms on CPU |
| Edge-TTS over local TTS | Coqui, Bark, XTTS | Edge-TTS is free, fast, high quality, supports 40+ languages |
| Corrected alignment over Whisper | Whisper forced alignment | Whisper base produces 43s drift on Vietnamese, corrected mode = 0.000s |
| PIL pre-compose over FFmpeg overlays | N-overlay FFmpeg filter chain | 12-overlay chain took 407s, PIL rawvideo pipe takes 14.8s |
| LLM extraction over RAG chunking | 500-char chunk + BM25 retrieve | Chunking destroys entity context, causes fact contamination |
| PCM concat over MP3 concat | FFmpeg `-c copy` on MP3 | MP3 frame alignment produces inter-chunk silence gaps |

---

## Limitations & Honest Assessment

### What this project IS
- A **fixed sequential pipeline** with LLM calls at each stage
- Each stage runs predetermined operations and passes output to the next
- Fallback chains (`try/except`) handle failures, not dynamic reasoning
- Two feedback loops exist: script lint -> rewrite, quality gate -> rewrite

### What this project is NOT
- Not a true **agent system** — stages don't decide what to do next based on observations
- No dynamic tool selection — the pipeline always runs the same 6 stages in the same order
- No autonomous reasoning loop — the system doesn't evaluate "should I re-research or continue?"
- The `agent/` directory naming is a convention, not a description of behavior

### What would make it agentic
- Research stage that autonomously decides "not enough facts, let me try different queries"
- Dynamic routing: "this topic needs Reddit threads, not Wikipedia"
- Self-evaluation: "this script doesn't match the topic well, let me re-research instead of just rewriting"
- Planning stage that adapts strategy based on what research found

### Known technical limitations
- Gemma 4B hallucination on sparse topics (anti-hallucination prompt mitigates but doesn't eliminate)
- Generic-word entity names (e.g., "Power", "Black Holes") are harder for topic entity matching
- Vietnamese celebrity topics have lower fact relevance due to common-word entity names
- Image search quality depends heavily on how specific the `image_keywords` are
