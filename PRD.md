# PRD — Full Implementation Roadmap

Planning document for all improvements identified in ARCHITECTURE.md.
No code here — this is a sequenced implementation plan with goals, approach, dependencies, effort, and acceptance criteria per feature.

---

## Phasing Overview

| Phase | Theme | Effort | Requires |
|---|---|---|---|
| 1 | Pipeline Robustness | 3–5 days | Nothing new |
| 2 | Content Quality | 4–6 days | Nothing new |
| 3 | Agentic RAG | 7–10 days | Vector store, embedding model |
| 4 | Trending Enhancement | 5–7 days | Phase 3 infra |

Each phase is self-contained and ships value independently. Earlier phases de-risk later ones.

---

## Phase 1 — Pipeline Robustness

Goal: make the existing pipeline more reliable and more transparent to the user. No new infra needed — just code-level fixes.

---

### 1.1 Circuit Breaker for LLM Providers

**Problem today:** When Groq is down, every LLM call still tries Groq first, waits through all retries (up to 45 seconds), then falls to Gemini. A 10-block pipeline can stall for 7+ minutes on a degraded Groq endpoint.

**Proposed approach:**
- Track consecutive failures per provider in a session-level counter (stored in memory, not disk).
- After 3 consecutive failures from the same provider, flip the provider order for the rest of the session.
- Reset the counter when a call succeeds.
- Surface provider health in the web UI status bar as a colored dot (green / yellow / red).

**Dependencies:** None — pure logic change in `src/llm_client.py`.

**Effort:** 1 day

**Acceptance criteria:**
- After 3 consecutive failures from provider A, the next call tries provider B first
- Counter resets to 0 on the next successful response from provider A
- UI status bar shows green/yellow/red for each provider
- No change to the existing retry-with-backoff behavior within a single call

---

### 1.2 Plan Review Step in UI

**Problem today:** The plan is invisible. User types a prompt, waits, and has no idea what topic, language, queries, and skill the system chose until the video is done.

**Proposed approach:**
- After the plan agent finishes, pause the pipeline and emit a "plan ready" SSE event containing: topic, language, style, skill chosen, and the search queries list.
- UI shows a "Review Plan" modal/panel with those fields — all editable.
- User clicks "Approve & Continue" to resume, or edits fields and clicks "Revise".
- On revise, the edited values are sent back as explicit user overrides and the pipeline continues (no second plan LLM call needed).

**Dependencies:** None — extends existing SSE event stream and `AgentConfig` override system.

**Effort:** 2 days (1 day backend pause/resume, 1 day UI panel)

**Acceptance criteria:**
- Pipeline pauses after plan stage and shows plan data in the UI
- User can edit topic, language, skill, and queries in the modal
- Edited values are used for research — verified by checking `plan.json` artifact
- "Approve without changes" continues immediately with the original plan
- If user closes the modal without action, pipeline resumes after 60 seconds (timeout fallback)

---

### 1.3 Per-Skill Quality Gate Questions

**Problem today:** The same 5 questions are used for every skill type. A `comedy` script is judged on "lore depth" and "escalating tension", which don't apply. A `comparison` script is judged on "hidden details" when it should be judged on fairness and clarity.

**Proposed approach:**
- Define a question set per skill category in each `skills/*.json` file under a new `"quality_gate_questions"` key.
- Fall back to the current 5 default questions if `quality_gate_questions` is missing.
- The quality gate agent reads the active skill's questions instead of using the hardcoded prompt.

**Question sets to write (examples):**
- `comedy`: Does the first line land an absurd premise? Is at least one moment genuinely unexpected?
- `comparison`: Is the comparison fair and evidence-based? Does each side get a clear win condition?
- `dark_secrets`: Does each revelation feel genuinely hidden rather than well-known? Is there a "why was this suppressed" angle?
- `theory`: Does the theory have internal consistency? Is the evidence chain clear?
- `lore_deep_dive`: Is the world-building detail specific enough to satisfy a lore fan? Is there a "so what" conclusion?

**Dependencies:** None.

**Effort:** 1.5 days (0.5 to extend quality gate loader, 1 to write question sets for all 13 skills)

**Acceptance criteria:**
- Each `skills/*.json` file has a `quality_gate_questions` array with 5 questions
- `quality_gate.py` uses the skill's questions when available
- Running quality gate on a comedy script does not ask "does it reveal hidden lore"
- Fallback to default 5 questions when skill has no `quality_gate_questions` key

---

### 1.4 CPU Encode Fallback (x264)

**Problem today:** The editor calls FFmpeg with NVENC hardware encoding. On machines without an NVIDIA GPU, every render fails immediately.

**Proposed approach:**
- Before starting a render, probe whether NVENC is available using `ffmpeg -encoders`.
- If NVENC is not available, fall back to `libx264` with `preset fast crf 23`.
- Log which encoder was selected.
- Add a config flag in `profiles/default.json` under `editor` to force CPU encode even when GPU is available (useful for headless servers).

**Dependencies:** FFmpeg already installed — no new deps.

**Effort:** 0.5 days

**Acceptance criteria:**
- On a machine without NVENC, render completes using libx264
- On a machine with NVENC, render uses NVENC (no regression)
- `profiles/default.json` has `force_cpu_encode: false` flag
- When `force_cpu_encode: true`, libx264 is used regardless of GPU availability

---

### 1.5 Lint Rewrite Loop (Multi-attempt)

**Problem today:** The script agent rewrites only once if lint fails. If the second attempt also scores poorly, the better of two mediocre scripts ships.

**Proposed approach:**
- Allow up to 3 lint rewrite attempts (configurable in profile as `max_lint_rewrites`).
- Track the best-scoring script across all attempts.
- Stop early if score reaches a "good enough" threshold (configurable as `lint_accept_score`).

**Dependencies:** None.

**Effort:** 0.5 days

**Acceptance criteria:**
- `profiles/default.json` has `max_lint_rewrites: 3` and `lint_accept_score: 80` under `agent`
- Script agent runs up to 3 rewrites and keeps the highest-scoring result
- Stops early when score >= `lint_accept_score`
- Logs each attempt's score

---

## Phase 2 — Content Quality

Goal: improve the quality of research and images. No new infra — extends existing pipelines.

---

### 2.1 Wikipedia Commons as Image Source

**Problem today:** Image search is DuckDuckGo-only. If DDG blocks the request or returns irrelevant results, blocks have zero images.

**Proposed approach:**
- Add `src/images/wikimedia_source.py` that queries the Wikimedia Commons API (`commons.wikimedia.org/w/api.php?action=query&list=search`).
- Use it as a fallback source when DuckDuckGo returns fewer than N images for a block.
- Filter out SVGs and images smaller than 400px on the short side.
- Prefer images tagged as "photograph" over diagrams/artwork for background mode.

**Dependencies:** Wikimedia Commons has an open JSON API, no key needed.

**Effort:** 1.5 days

**Acceptance criteria:**
- `get_images_for_script` tries Wikimedia Commons when DDG returns < 3 images for a block
- Wikimedia results go through the same quality filter as DDG results
- SVGs and tiny images are excluded
- Wikimedia Commons is not called when DDG already found enough images (no wasted requests)

---

### 2.2 Research Gap-Fill Multiple Iterations

**Problem today:** The reflect/fill loop runs exactly once. If there are still gaps after the first fill, research ends with missing angles.

**Proposed approach:**
- Allow the gap-fill loop to run up to `max_research_iterations` times (already a profile key, currently defaults to 1).
- Each iteration avoids domains already visited in previous iterations.
- Stop early when fact count reaches `max_final_facts`.
- Surface the iteration count in the run artifact (`research.json`) for debugging.

**Dependencies:** None — `max_research_iterations` is already a profile key.

**Effort:** 0.5 days

**Acceptance criteria:**
- `max_research_iterations: 3` in profile causes up to 3 gap-fill iterations
- Each iteration's visited domains are excluded from the next
- Research stops when `max_final_facts` is reached even if iterations remain
- `research.json` artifact records how many iterations were run

---

### 2.3 BM25 Threshold Tuning + Soft Floor

**Problem today:** BM25 threshold of 1.5 is hardcoded and too aggressive — drops useful pages that mention the topic indirectly.

**Proposed approach:**
- Move threshold to profile (`bm25_threshold`, already a profile key at 1.5).
- Add a soft floor: if fewer than 3 pages pass the threshold, lower it by 0.3 and retry scoring (up to 2 steps down).
- This ensures research always has at least 3 pages to extract from, even for niche topics.

**Dependencies:** None.

**Effort:** 0.5 days

**Acceptance criteria:**
- `bm25_threshold` in profile controls the initial threshold
- If fewer than 3 pages pass, threshold is lowered by 0.3 and pages are re-scored
- At most 2 threshold reductions before giving up
- Threshold reduction is logged with a warning

---

### 2.4 Fallback TTS Engine (ElevenLabs or Kokoro)

**Problem today:** Edge-TTS is the only engine. If Microsoft throttles it, there is no fallback and the render fails.

**Proposed approach:**
- Define a TTS provider interface: `synthesize(text, voice, rate, pitch) -> (audio_bytes, word_timings)`.
- Implement `EdgeTTSProvider` (current behavior, wrapped).
- Implement `KokoroProvider` as a local fallback — Kokoro runs offline via a Python package, no API key needed.
- Add `tts_providers: ["edge", "kokoro"]` to profile — try in order, fall back on error.
- Word timings from Kokoro are approximated (character-count-based) since Kokoro doesn't emit boundary events.

**Dependencies:** `kokoro` Python package (local inference, ~500MB model download on first use).

**Effort:** 2 days (1 day interface + Kokoro, 0.5 day Edge wrapper, 0.5 day fallback wiring)

**Acceptance criteria:**
- `tts_providers: ["edge", "kokoro"]` in profile controls fallback order
- If Edge-TTS fails, Kokoro synthesizes the same text without crashing the pipeline
- Kokoro word timings are within 10% of actual durations (good enough for subtitle sync)
- First Kokoro call downloads the model and caches it — subsequent calls skip download
- No regression when Edge-TTS works normally

---

### 2.5 BGM Preview and Manual Track Selection

**Problem today:** BGM is a random pick based on mood. User has no way to preview or select a specific track before rendering.

**Proposed approach:**
- Add a BGM preview panel in the Agent UI showing the current mood selection and a list of available tracks.
- Each track row has a play/stop button (HTML5 `<audio>` element, served from `/assets/audio/bgm/`).
- User can pin a specific track — pinning sends `bgm_track` in the POST body.
- `VideoManager` uses the pinned track path instead of random selection when `bgm_track` is set.

**Dependencies:** None — tracks already exist in `assets/audio/bgm/`.

**Effort:** 1 day

**Acceptance criteria:**
- BGM preview panel lists all tracks in `assets/audio/bgm/` grouped by inferred mood
- Each track can be played/stopped in browser without leaving the page
- Pinning a track sends `bgm_track` filename in agent POST body
- When `bgm_track` is set, that exact file is used in the render
- When not pinned, random mood-based selection still works

---

## Phase 3 — Agentic RAG for Research

Goal: replace the current BM25-only research pipeline with a hybrid semantic retrieval system that avoids re-fetching previously researched topics.

This is the largest architectural change. It introduces a local vector store and embedding model as new infrastructure.

---

### 3.1 Infrastructure: Vector Store + Embedding Model

**Problem today:** No vector search capability. Research always starts from scratch even for topics covered last week.

**Proposed approach:**
- Use **ChromaDB** as the local vector store (file-backed, zero config, pure Python).
- Use **nomic-embed-text** via Ollama as the embedding model (local inference, no API key, ~270MB).
- Store one collection per research session, keyed by topic + date range.
- Vector store lives at `.vector_store/` in the project root.

**Alternative if Ollama is unavailable:** Fall back to TF-IDF vectors (much lower quality but zero new deps).

**Dependencies:** `chromadb` pip package, Ollama installed locally with `nomic-embed-text` pulled.

**Effort:** 1 day (setup, wiring, fallback path)

**Acceptance criteria:**
- `chromadb` importable in `.venv`
- Embedding a string returns a consistent 768-dim vector
- ChromaDB collection persists to `.vector_store/` and survives restart
- If Ollama is not running, system falls back to TF-IDF without crashing

---

### 3.2 Semantic Chunking of Crawled Pages

**Problem today:** Each crawled page is sent as a single blob (first 3000 chars) to the LLM. The most specific facts are often cut off.

**Proposed approach:**
- After crawling, split each page into semantic chunks by heading or paragraph boundary (max ~400 tokens per chunk).
- Add a context prefix to each chunk: the page title, section heading, and a 1-sentence summary of the full page.
- This "context enrichment" ensures each chunk is self-contained — the LLM extracting from chunk 7 knows it came from section "Episode 3 controversy" of a Crunchyroll article.
- Store chunks individually in the vector store with metadata: `{page_url, section, chunk_index, topic}`.

**Dependencies:** Phase 3.1 (vector store).

**Effort:** 1.5 days

**Acceptance criteria:**
- A 5000-char page produces 10–15 chunks, not one blob
- Each chunk has a context prefix (page title + section + summary)
- All chunks for a crawl session are stored in ChromaDB with correct metadata
- Total tokens sent to extraction LLM is equal to or less than before (chunk is shorter, but better targeted)

---

### 3.3 Hybrid Search (BM25 + Vector)

**Problem today:** BM25-only search misses semantically related pages that don't contain exact keyword matches.

**Proposed approach:**
- On each research query, run two searches in parallel:
  1. BM25 keyword search (existing)
  2. Vector similarity search against the ChromaDB collection
- Combine the ranked lists using **Reciprocal Rank Fusion (RRF)**: score = 1/(k + rank_bm25) + 1/(k + rank_vector) where k=60.
- The fused list replaces the current BM25-only ranked list.
- Only chunks already in the vector store are eligible for vector search — new pages go through BM25 first, then get embedded and stored.

**Dependencies:** Phase 3.1 + 3.2.

**Effort:** 1 day

**Acceptance criteria:**
- A query for "Naruto Shippuden fillers" retrieves chunks even from pages that say "filler arcs" not "fillers"
- RRF scores are computed correctly (verified by unit test with known rankings)
- When vector store is empty, system falls back to BM25-only (no crash)

---

### 3.4 Parent Document Retrieval

**Problem today:** Even with chunking, the LLM only sees the matching chunk — it loses context from surrounding paragraphs.

**Proposed approach:**
- When a chunk ranks highly in hybrid search, fetch its parent page from the vector store metadata.
- Send the top 2–3 chunks from the same parent page together as a "document window" to the extraction LLM, rather than individual chunks.
- Cap the document window at `page_text_max_chars` to stay within LLM context limits.

**Dependencies:** Phase 3.2 (chunk metadata includes parent page URL).

**Effort:** 0.5 days

**Acceptance criteria:**
- Top-ranked chunk triggers retrieval of up to 2 sibling chunks from same page
- Document window is deduplicated (no repeated text)
- Window is truncated to `page_text_max_chars` if siblings push it over the limit

---

### 3.5 Research Semantic Cache

**Problem today:** Re-researching "Naruto lore" for a new video re-fetches and re-extracts all the same pages from scratch.

**Proposed approach:**
- Before starting web search, embed the topic + queries and run a cosine similarity search against a persistent "research cache" ChromaDB collection.
- If similarity > 0.85, retrieve the cached facts directly and skip web fetch + extraction.
- Cache entries expire after 7 days (stored as metadata timestamp).
- Cache miss or expired entry: run full research pipeline, then store results in cache for next time.

**Dependencies:** Phase 3.1, Phase 3.3.

**Effort:** 1.5 days

**Acceptance criteria:**
- Running research on "Naruto vs Sasuke fight analysis" twice within 7 days hits the cache on the second run
- Cache hit skips DuckDuckGo + crawl + extraction (verified by checking that no HTTP requests are made)
- Cache entries older than 7 days are ignored (re-research happens)
- When cache is bypassed (forced refresh flag), full pipeline runs normally

---

## Phase 4 — Trending Enhancement

Goal: make the trending agent faster, smarter, and extensible to categories beyond anime.

---

### 4.1 Semantic Cache for Trending Brainstorm

**Problem today:** Every trending refresh calls the 70b LLM for each of the top 8 shows, even if we brainstormed angles for 6 of them last week.

**Proposed approach:**
- After fetching and scoring shows, summarize each show's API data into a dense paragraph using a fast LLM (8b, single call).
- Embed the summary vector.
- Compare against a persistent "trending brainstorm cache" collection (ChromaDB, keyed by show summary embedding).
- Cache hit (cosine > 0.90): reuse cached angle, skill, hook, queries. Skip 70b call.
- Cache miss: run 70b brainstorm as today, store result in cache.
- Cache entries expire after 7 days. Urgency ≥ 1.0 (hot episode drop) always bypasses cache.

**Dependencies:** Phase 3.1 (ChromaDB infra). 70b brainstorm still needed for cache misses.

**Effort:** 1.5 days

**Acceptance criteria:**
- Second trending refresh within 7 days for the same shows skips the 70b LLM call for cached entries
- Cache hit verified by log message "trending cache hit for: {title}"
- Hot episode drop (urgency ≥ 1.0) always brainstorms fresh — not served from cache
- Total refresh time with 6/8 cache hits is < 15 seconds (vs 45+ seconds today)

---

### 4.2 Agentic Brainstorm with Tools

**Problem today:** The brainstorm LLM only has the show synopsis and Reddit context available. It can't look up current Reddit discussion or fetch lore details before committing to an angle.

**Proposed approach:**
- Give the brainstorm LLM two tool calls it can make before returning the angle:
  1. `search_reddit(title)` — fetches the top 5 posts from r/anime for that title
  2. `get_anilist_lore(id)` — fetches AniList tags, relations, and staff data
- The LLM decides whether to call them (tool use is optional per call).
- If a tool call is made, the result is appended to the brainstorm context.
- Max 2 tool calls per brainstorm to bound latency.

**Dependencies:** Existing Reddit fetch and AniList fetch functions can be reused.

**Effort:** 1.5 days

**Acceptance criteria:**
- Brainstorm prompt includes tool call schema
- LLM optionally calls `search_reddit` or `get_anilist_lore` before returning angle
- Tool results are logged in `trend_log.json` for debugging
- If tools are unavailable (network error), brainstorm continues without them
- Max 2 tool calls enforced — LLM cannot loop indefinitely

---

### 4.3 Second Trending Category — Gaming (IGDB)

**Problem today:** Only anime is implemented. Selecting any other category silently returns nothing.

**Proposed approach:**
- Add `src/content_sources/game_trends.py` that queries the **IGDB API** (Twitch-owned, free tier available) for games released or updated in the last 30 days, sorted by hype score.
- Apply the same scoring model as anime trends: trending score + recency bonus.
- Add "gaming" to the `trendCategory` select in the UI.
- The brainstorm skill guide already has `top_list`, `explained`, `theory`, `comparison` — no new skills needed.

**Dependencies:** IGDB API key (free, requires Twitch developer account). New `IGDB_CLIENT_ID` and `IGDB_CLIENT_SECRET` env vars.

**Effort:** 2 days

**Acceptance criteria:**
- `IGDB_CLIENT_ID` + `IGDB_CLIENT_SECRET` in `.env` enable game trends
- Selecting "Gaming" in the UI category dropdown loads trending game cards
- Game cards show title, trending reason, and hook idea (same card format as anime)
- If IGDB keys are missing, gaming category shows "API key not configured" message
- Anime category still works unchanged

---

## Dependency Graph

```
Phase 1 items → independent, can ship in any order
Phase 2 items → independent, can ship in any order
Phase 3.1 (Vector Store) → must ship before 3.2, 3.3, 3.4, 3.5, 4.1
Phase 3.2 (Chunking) → must ship before 3.3, 3.4
Phase 3.3 (Hybrid Search) → must ship before 3.4
Phase 3.4 (Parent Retrieval) → standalone after 3.3
Phase 3.5 (Research Cache) → requires 3.1 and 3.3
Phase 4.1 (Trending Cache) → requires 3.1
Phase 4.2 (Agentic Brainstorm) → independent of Phase 3
Phase 4.3 (Gaming) → independent, just needs IGDB key
```

---

## Total Effort Estimate

| Phase | Items | Effort |
|---|---|---|
| Phase 1 — Robustness | Circuit breaker, Plan review, Per-skill QA, CPU fallback, Lint loop | 5.5 days |
| Phase 2 — Content Quality | Wikipedia Commons, Gap-fill loop, BM25 tuning, TTS fallback, BGM preview | 5.5 days |
| Phase 3 — Agentic RAG | Vector store, Chunking, Hybrid search, Parent retrieval, Research cache | 5.5 days |
| Phase 4 — Trending | Trending cache, Agentic brainstorm, Gaming category | 5 days |
| **Total** | | **~21 developer-days** |

---

## What's Already Done (Not Included Above)

These items from the original roadmap are already implemented:

- Quality gate N/5 score shown in UI pipeline
- Subtitle preset picker (minimal / energetic / cinematic)
- Hard stop on empty image blocks with clear error
- Jikan timeout increased to 10s with retry
- `hook_idea` and `trending_reason` displayed on trending cards
- Plan stage upgraded to llama-3.3-70b
