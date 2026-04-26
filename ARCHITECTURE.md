# Video Maker — Architecture & Flow

Full picture of how the system works today, what's broken, and where it's going.

---

## 1. Full Pipeline (Bird's Eye View)

```mermaid
flowchart TD
    UI["Web UI — Agent Tab"] -->|"POST /api/agent/run"| CORE["VideoAgent.run"]
    TREND["Trending Panel"] -->|"POST /api/trending/generate"| CORE

    CORE --> PLAN["Plan Agent"]
    PLAN --> RESEARCH["Research Agent"]
    RESEARCH --> SCRIPT["Script Agent"]
    SCRIPT --> QG["Quality Gate"]
    QG --> IMAGE["Image Agent"]
    IMAGE --> TTS["TTS Engine"]
    TTS --> EDITOR["Editor and FFmpeg"]
    EDITOR --> VIDEO[("Final Video MP4")]

    PLAN -->|"plan.json"| DISK[("Saved to disk\nruns/topic_timestamp/")]
    RESEARCH -->|"research.json"| DISK
    SCRIPT -->|"script.json"| DISK
    IMAGE -->|"script_final.json"| DISK

    style CORE fill:#1e1e28,stroke:#f59e0b,color:#e8eaf0
    style VIDEO fill:#052e16,stroke:#86efac,color:#86efac
    style DISK fill:#1e1e28,stroke:#6b7280,color:#6b7280
```

---

## 2. Plan Agent

Turns a free-text prompt into a structured plan the rest of the pipeline reads.

```mermaid
flowchart TD
    PROMPT["User types a prompt"] --> LLM["Small LLM decides the plan\nllama-3.1-8b — fast but limited"]
    LLM -->|"Understood"| PARSE["Read topic, language, style\nsearch queries, blocks, voice"]
    LLM -->|"Failed to parse"| HEURISTIC["Keyword fallback\nguess from words in the prompt"]
    PARSE --> MERGE["Apply any user overrides\ne.g. forced skill or voice"]
    HEURISTIC --> MERGE
    MERGE --> OUT["Ready plan\ntopic, language, style, queries\nblocks count, voice, BGM mood"]

    style LLM fill:#1e1e28,stroke:#38bdf8,color:#e8eaf0
    style HEURISTIC fill:#1e1e28,stroke:#f59e0b,color:#e8eaf0
```

**What the plan contains:**
- Topic, language (English / Vietnamese), style, topic category
- Up to 8 search queries for research
- Number of script blocks (3–10), image display mode, voice, BGM mood

**⚠ Known problems:**
- The 8b model is too small — it misreads complex prompts and gets language wrong for mixed-language topics
- If LLM fails, the keyword fallback runs silently with no warning to the user
- The plan is invisible — user can't review or adjust queries before research starts

**💡 Improvements:**
- Use the 70b model for planning, or enforce a structured JSON output schema so parsing never fails
- Show the plan in the UI as a "confirm before research" step — let the user edit queries
- Surface a warning banner if the fallback was used

---

## 3. Research Agent

Gathers facts from the web. Three stages: search → crawl → extract.

### How it works today

```mermaid
flowchart TD
    IN["Plan: topic + search queries"] --> DDG["Search DuckDuckGo\nup to 12 results per query"]
    IN --> WIKI["Fetch Wikipedia\nin target language and English"]
    DDG --> RANK["Rank URLs by relevance\nBM25 keyword scoring, threshold 1.5"]
    WIKI --> RANK
    RANK -->|"Best 6 pages"| CRAWL["Download each page\nkeep first 3000 characters"]
    CRAWL --> EXTRACT["Ask LLM to extract facts\none call per page, runs in parallel"]
    EXTRACT -->|"Page not relevant"| SKIP["Discard this page"]
    EXTRACT -->|"Facts found"| DEDUP["Remove near-duplicate facts\nJaccard word overlap above 50%"]
    DEDUP --> CHECK{"Do we have enough facts?"}
    CHECK -->|"Yes, or already tried once"| DONE["Final fact list\nup to 12 facts"]
    CHECK -->|"No — gaps detected"| FILL["Search for missing angles\navoid domains already visited"]
    FILL --> CRAWL
    DONE -->|"Still empty"| FALLBACK["Ask LLM to generate facts directly\nno web source, low confidence"]
    FALLBACK --> DONE

    style EXTRACT fill:#1e1e28,stroke:#38bdf8,color:#e8eaf0
    style FALLBACK fill:#1e1e28,stroke:#f59e0b,color:#e8eaf0
    style SKIP fill:#450a0a,stroke:#fca5a5,color:#fca5a5
```

**⚠ Known problems:**
- Gap-filling loop only runs once — misses multiple research holes in one pass
- Pages are cut off at 3000 characters — the most specific facts are often in the second half of an article
- BM25 threshold 1.5 is too aggressive — drops useful pages that mention the topic indirectly
- LLM fallback makes up facts with no source URL and a low confidence score of 0.3
- Wikipedia always fetches English even when the topic is already in English (wasted API call)
- Hard cap of 12 facts — fine for simple topics, too few for dense lore

### Proposed future architecture — Agentic RAG with Vector Store

```mermaid
flowchart TD
    IN["Plan: topic + base queries"] --> EXPAND["Expand queries\nGenerate 3 rephrasings using a fast LLM"]

    EXPAND --> CACHE{"Check vector cache\nHave we researched this before?"}
    CACHE -->|"Yes — high confidence match"| SYNTH["Synthesise directly\nfrom cached chunks"]
    CACHE -->|"No — research from scratch"| SEARCH["Search web\nDuckDuckGo and Wikipedia"]

    SEARCH --> RANK["Rank URLs by relevance\nBM25 scoring"]
    RANK --> CRAWL["Download pages\nCrawl4AI"]
    CRAWL --> CHUNK["Split into semantic chunks\nby heading or paragraph"]
    CHUNK --> ENRICH["Add page summary to each chunk\nso context is never lost"]
    ENRICH --> EMBED["Embed chunks\nfast local model e.g. nomic-embed-text"]
    EMBED --> STORE[("Vector and keyword index\nchunk text + parent page mapping")]

    EXPAND -.->|"All query variants"| HYBRID["Hybrid search\nvector similarity + BM25 keyword"]
    STORE --> HYBRID
    HYBRID --> FUSE["Fuse rankings\nReciprocal Rank Fusion"]
    FUSE --> PARENT["Fetch full parent page\nfor each top chunk"]

    PARENT --> EXTRACT["Extract atomic facts\nsmall fast LLM, strict format"]
    EXTRACT --> DEDUP["Remove duplicate facts\nword-level Jaccard"]
    DEDUP --> SYNTH

    SYNTH --> EVAL{"Enough coverage?"}
    EVAL -->|"Yes"| OUT["Final facts ready"]
    EVAL -->|"No"| FILL["Search for missing angles\nor relax similarity threshold"]
    FILL --> EXPAND

    style EXPAND fill:#1e1e28,stroke:#38bdf8,stroke-width:2px,color:#e8eaf0
    style HYBRID fill:#1e1e28,stroke:#ef4444,stroke-width:2px,color:#e8eaf0
    style STORE fill:#1e1e28,stroke:#f59e0b,color:#e8eaf0
    style EXTRACT fill:#1e1e28,stroke:#10b981,color:#e8eaf0
    style SYNTH fill:#1e1e28,stroke:#8b5cf6,color:#e8eaf0
    style PARENT fill:#1e1e28,stroke:#f43f5e,stroke-width:2px,color:#e8eaf0
    style ENRICH fill:#1e1e28,stroke:#eab308,stroke-width:2px,color:#e8eaf0
```

**What this solves vs today:**
- Semantic cache avoids re-researching the same topic (e.g. Naruto lore fetched last week)
- Contextual chunk enrichment means each chunk knows which page and section it came from — no more truncation blindness
- Hybrid search recalls facts that keyword-only BM25 misses
- Parent document retrieval gives the LLM full context, not just the matching snippet
- Reflection loop can run multiple times and lower the threshold instead of giving up

---

## 4. Script Agent

Turns the fact list into a narrated script, then checks and optionally rewrites it.

```mermaid
flowchart TD
    FACTS["Fact list from research"] --> SKILL["Pick the best skill template\nBM25 match against 13 templates"]
    SKILL -->|"Good match found"| SELECTED["Use chosen skill\ne.g. did_you_know, comparison, theory"]
    SKILL -->|"No good match"| DEFAULT["Use default template"]
    SELECTED --> WRITE["Write the script\nLarge LLM, 70b model"]
    DEFAULT --> WRITE
    WRITE --> LINT["Run linter\ncheck for weak openers, bad endings"]
    LINT -->|"Passes"| OUT["Script ready\nblocks, lint score"]
    LINT -->|"Fails"| REWRITE["Rewrite once with linter feedback"]
    REWRITE --> LINT2["Lint again"]
    LINT2 -->|"Keep whichever scored higher"| OUT

    OUT --> IMGMODE{"How should images display?"}
    IMGMODE -->|"User chose explicitly"| KEEP["Use user's choice"]
    IMGMODE -->|"Auto"| INFER["Count visual signals in the text\nlandscape words vs action words"]
    INFER --> FINAL["Script with image mode set"]
    KEEP --> FINAL

    style WRITE fill:#1e1e28,stroke:#38bdf8,color:#e8eaf0
    style REWRITE fill:#1e1e28,stroke:#f59e0b,color:#e8eaf0
```

**⚠ Known problems:**
- Rewrite happens only once — if the second attempt is also bad, the better of two mediocre scripts ships
- Image mode inference counts keywords naively — "ocean" in a political script still triggers background mode
- Vietnamese validation rejects scripts with English proper nouns (character names, titles)
- No tracking of which facts were actually used — high-score facts can be silently ignored

**💡 Improvements:**
- Show lint issues in the UI so the user can fix them manually
- Replace image mode inference with an explicit toggle in the UI
- Log unused facts so the user can see what got left on the cutting room floor

---

## 5. Quality Gate

Five yes/no questions asked by a judge LLM. Script rewrites if it fails.

```mermaid
flowchart TD
    SCRIPT["Script from Script Agent"] --> JUDGE["Judge LLM scores the script\n5 yes/no questions, 70b model"]
    JUDGE --> COUNT{"3 or more yes answers?"}
    COUNT -->|"Yes"| PASS["Script approved"]
    COUNT -->|"No — with feedback"| REWRITE["Rewrite the weakest part\none attempt"]
    COUNT -->|"No — no feedback"| WARN["Log a warning and continue anyway"]
    REWRITE --> KEEP["Keep whichever version scored higher"]
    KEEP --> PASS
    WARN --> PASS

    JUDGE -.->|"No API key configured"| SKIP["Skip gate entirely\nno check at all"]

    style JUDGE fill:#1e1e28,stroke:#38bdf8,color:#e8eaf0
    style SKIP fill:#450a0a,stroke:#fca5a5,color:#fca5a5
    style PASS fill:#052e16,stroke:#86efac,color:#86efac
```

**The 5 questions:**
1. Does the hook open with a shocking fact, bold claim, or curiosity gap?
2. Does it reveal hidden details rather than summarise basic plot?
3. Is there at least one fact a knowledgeable fan wouldn't already know?
4. Does the last sentence loop back to the opening for replay?
5. Does tension escalate — each revelation more surprising than the last?

**⚠ Known problems:**
- Threshold of 3/5 means a script can fail two questions and still ship
- The judge's feedback is never shown to the user
- Gate skips entirely when no API key is set — no warning, no fallback check
- Same five questions for all skill types — a comedy script shouldn't be judged on lore depth

**💡 Improvements:**
- Show the N/5 score in the UI progress bar (green above 4, yellow at 3, red below)
- Surface the weakest-part feedback so the user can manually fix it
- Define per-skill question sets — comedy checks absurdity and timing, dark_secrets checks reveal depth

---

## 6. Image Agent

Finds and attaches images to each script block.

```mermaid
flowchart TD
    BLOCKS["Script blocks with image keywords"] --> ENRICH["Anchor keywords to the topic\ne.g. fight scene becomes Naruto fight scene"]
    ENRICH --> SEARCH["Search DuckDuckGo Images\nup to 30 images per block"]
    SEARCH --> MAP["Attach image paths to each block"]
    MAP -->|"Block has images"| OK["script_final.json with images"]
    MAP -->|"Block has no images"| WARN["Log a warning\nvideo will have black frames here"]
    WARN --> OK

    style SEARCH fill:#1e1e28,stroke:#38bdf8,color:#e8eaf0
    style WARN fill:#450a0a,stroke:#fca5a5,color:#fca5a5
```

**⚠ Known problems:**
- Only searches DuckDuckGo — no fallback if DDG blocks the request or returns irrelevant results
- No quality filter — blurry, watermarked, and portrait images all pass through
- Empty image blocks produce black frames in the video instead of stopping with an error
- Downloads 30 images per block but the editor uses only a few — wasteful

**💡 Improvements:**
- Add Wikipedia Commons as a second source for historical and factual topics
- Hard-stop the pipeline when a block has zero images — surface a clear error to the user
- Filter out portrait images when display mode is background

---

## 7. TTS Engine

Converts script text to speech in parallel chunks, then corrects timing drift.

```mermaid
flowchart TD
    TEXT["Script block text"] --> SPLIT["Split into chunks\nmax 500 characters each\ncut at sentence boundaries"]
    SPLIT -->|"Multiple chunks"| PARALLEL["Synthesise chunks in parallel\nEdge-TTS, max 5 at once"]
    SPLIT -->|"Short text"| SINGLE["Synthesise as one call"]
    PARALLEL --> MERGE["Concatenate audio\nno silence between chunks"]
    SINGLE --> TIMING["Extract word timestamps\nfrom Edge-TTS boundary events"]
    MERGE --> TIMING
    TIMING --> DRIFT{"Is timing off by more than 160ms?"}
    DRIFT -->|"Yes"| FIX["Stretch or compress the timeline\nscale 0.88x to 1.18x"]
    DRIFT -->|"No"| DONE["Word timings ready"]
    FIX --> DONE
    DONE --> CACHE["Save to cache\nSHA256 hash of text and settings"]

    style PARALLEL fill:#1e1e28,stroke:#38bdf8,color:#e8eaf0
    style FIX fill:#1e1e28,stroke:#f59e0b,color:#e8eaf0
```

**⚠ Known problems:**
- Max concurrent chunks (5) is hardcoded — should be in the profile
- Edge-TTS is the only engine — if Microsoft throttles it, there is no fallback
- Switching alignment mode doesn't invalidate the cache — stale results can be used
- Drift correction range (0.88–1.18x) may not be wide enough for very long blocks

**💡 Improvements:**
- Move the concurrency limit to `profiles/default.json`
- Add ElevenLabs or Kokoro as an optional fallback TTS engine
- Include alignment mode in the cache key

---

## 8. Editor and Renderer

Composes subtitles, images, audio, and background into a final video.

```mermaid
flowchart TD
    IMGS["Images and script blocks"] --> CAPTIONS["Build subtitle captions\nmax 30 characters per line\nmax 15 characters per second"]
    WTIMES["Word timestamps from TTS"] --> CAPTIONS
    CAPTIONS --> SMOOTH["Smooth timing gaps\nmin 60ms, max 240ms between captions"]
    SMOOTH --> ASS["Write subtitle file\nASS format"]

    IMGS --> COMPOSE["Pre-compose images in Python\nPIL, piped as raw video to FFmpeg"]
    ASS --> FFMPEG["FFmpeg encodes the video\nNVENC hardware GPU encoder\nKen Burns pan-zoom on images\noverlays and BGM mixed in"]
    COMPOSE --> FFMPEG
    FFMPEG --> OUT["output.mp4"]

    style FFMPEG fill:#1e1e28,stroke:#38bdf8,color:#e8eaf0
    style OUT fill:#052e16,stroke:#86efac,color:#86efac
```

**⚠ Known problems:**
- Requires an NVIDIA GPU — no CPU fallback if NVENC is unavailable
- BGM is a random pick based on mood — user can't choose a specific track or preview it
- Subtitle preset (minimal / energetic / cinematic) is locked to minimal — the other presets exist but aren't exposed in the UI

**💡 Improvements:**
- Add x264 CPU encode path as fallback when NVENC is not available
- Let user pick the subtitle preset in the Agent UI
- BGM preview panel before generating

---

## 9. Trend Discovery

Finds what anime is trending right now and brainstorms video angles for it.

### How it works today

```mermaid
flowchart TD
    PANEL["User opens Trending panel"] --> FETCH["Fetch from 3 sources in parallel"]
    FETCH --> AL["AniList — current season\nsorted by trending score"]
    FETCH --> JK["Jikan — current season\nMAL data"]
    FETCH --> RD["Reddit r/anime\nepisode discussion posts"]

    AL --> FILTER["Remove daily kids shows\nblocklist and episode count above 800"]
    FILTER --> SCORE["Score each show\nAniList 50% + Jikan 30% + Reddit 20%\n+15% bonus if episode aired in last 24 hours"]
    JK --> SCORE
    RD --> SCORE
    SCORE --> TOP8["Pick top 8 shows"]

    TOP8 --> BRAIN["Brainstorm a video angle per show\n70b LLM, given synopsis and Reddit context\noutput: angle, skill, hook, search queries"]
    BRAIN --> DEDUP{"Already made this video\nin the last 7 days?"}
    DEDUP -->|"Same show + skill + similar angle"| SKIP["Skip it"]
    DEDUP -->|"Episode just dropped — urgency 1.0"| ALWAYS["Always allow\neven if recently covered"]
    DEDUP -->|"New combination"| LOG["Record in trend log"]
    LOG --> CARDS["Show as cards in the UI\nclick to fill the prompt form"]
    ALWAYS --> CARDS

    style BRAIN fill:#1e1e28,stroke:#38bdf8,color:#e8eaf0
    style ALWAYS fill:#052e16,stroke:#86efac,color:#86efac
    style SKIP fill:#450a0a,stroke:#fca5a5,color:#fca5a5
```

**⚠ Known problems:**
- AniList seasonal trending still surfaces Precure and AiPri above mainstream shows — popularity is not considered
- Jikan and AniList classify multi-cour shows differently, so the MAL ID cross-match rarely works
- Reddit posts use romanized titles (e.g. Shingeki no Kyojin) while AniList uses English (Attack on Titan) — fuzzy match fails
- 5 second timeout is too short for Jikan which rate-limits aggressively
- Only anime is implemented — all other categories silently return nothing

### Proposed future architecture — Semantic Cache + Agentic Brainstorm

```mermaid
flowchart TD
    PANEL["User opens Trending panel"] --> FETCH["Fetch from 3 sources in parallel\nAniList, Jikan, Reddit"]
    FETCH --> FILTER["Filter and score shows\nsame scoring as today"]
    FILTER --> TOP8["Top 8 candidates"]

    TOP8 --> FAST_LLM["Summarise each show with a fast LLM\ncompress raw API data into a dense paragraph"]
    FAST_LLM --> EMBED["Embed the summary\nto a vector"]

    EMBED --> CACHE{"Have we seen this show before?\ncosine similarity above 0.90"}
    CACHE -->|"Yes — cache hit"| PULL["Reuse cached angle, skill, hook\nskip the expensive brainstorm call"]
    CACHE -->|"No — never seen this"| BRAINSTORM["Brainstorm with a large LLM\n70b model with tools\ncan call search_reddit and get_lore\noutput: angle, skill, hook, queries"]
    BRAINSTORM --> STORE["Save result to cache\nvector + outputs"]

    PULL --> LOG["Record in trend log\ndedup check still applies"]
    STORE --> LOG
    LOG --> CARDS["Show as cards in the UI"]

    style FAST_LLM fill:#1e3a8a,stroke:#60a5fa,color:#eff6ff
    style BRAINSTORM fill:#4c1d95,stroke:#a78bfa,color:#f5f3ff
    style CACHE fill:#065f46,stroke:#34d399,color:#ecfdf5
    style PULL fill:#065f46,stroke:#34d399,color:#ecfdf5
    style STORE fill:#065f46,stroke:#34d399,color:#ecfdf5
```

**What this solves vs today:**
- Fast LLM summarisation makes caching practical — compare embeddings, not raw API JSON
- Cache hit avoids the 70b brainstorm call for shows covered recently — much faster refresh
- Large LLM with tools can look up Reddit threads and wiki lore before committing to an angle — richer, more accurate hooks

**💡 Near-term improvements (without the full cache rebuild):**
- Use AniList `popularity` score as a tiebreaker when trending scores tie
- Increase Jikan timeout to 10s with one retry
- Surface `trending_reason` and `hook_idea` text on each card (currently hidden)
- Add a second category — gaming via IGDB or music via Billboard

---

## 10. LLM Routing

Every LLM call goes through a central client that picks the right model and handles failures.

```mermaid
flowchart LR
    CALL["Any pipeline stage calls chat_completion"] --> RESOLVE["Look up stage in profiles/default.json\ngets provider list and model names"]
    RESOLVE --> P1["Try the first provider"]
    P1 -->|"Success"| RESP["Return the response"]
    P1 -->|"Service unavailable 503"| BACK["Wait and retry up to 3 times"]
    P1 -->|"Rate limited 429"| WAIT["Wait 5, 10, or 15 seconds\nthen retry up to 3 times"]
    P1 -->|"Any other error"| P2["Try the second provider"]
    BACK --> P1
    WAIT --> P1
    P2 -->|"Success"| RESP
    P2 -->|"Fail"| ERR["Raise error — pipeline stops"]

    style RESP fill:#052e16,stroke:#86efac,color:#86efac
    style ERR fill:#450a0a,stroke:#fca5a5,color:#fca5a5
```

**Stage to model mapping:**

| Stage | Primary | Fallback | Model size |
|---|---|---|---|
| script, quality_gate, refine, research | Groq | Gemini | 70b |
| research_extract | Groq | Gemini | 70b / 27b |
| plan, crawl, interest_rank, bank_extract, research_eval | Groq only | — | 8b |
| trend_brainstorm | Groq only | — | 70b |

**⚠ Known problems:**
- If Groq rate-limits during research, each retry blocks the pipeline for 5–15 seconds before falling through to Gemini
- Gemini fallback uses `gemma-3-27b-it` which is bad at returning valid JSON — causes frequent parse failures
- No circuit breaker — if Groq is down, every call still tries Groq first and waits through all retries

**💡 Improvements:**
- Circuit breaker: after 3 consecutive failures, flip provider order for the rest of the session
- Swap `gemma-3-27b-it` for `gemini-2.0-flash` — same speed, dramatically better JSON compliance
- Show provider health (green / yellow / red) in the UI status bar

---

## 11. Full Data Flow — Stage by Stage

```mermaid
flowchart TD
    PROMPT(["User types a prompt"]) --> P["Plan Agent\nsmall 8b model"]
    P -->|"Structured plan\ntopic, queries, style, blocks"| R["Research Agent\nextract facts from the web"]
    R -->|"Up to 12 facts\nwith source URLs"| S["Script Agent\n70b writer with linting"]
    S -->|"Script blocks\nwith image mode"| QG["Quality Gate\n70b judge, 5 questions"]
    QG -->|"Approved or rewritten"| I["Image Agent\nsearch and attach images"]
    I -->|"Blocks with image paths"| TTS_S["TTS Engine\nEdge-TTS parallel chunks"]
    TTS_S -->|"Audio file\nword timestamps"| ED["Editor\nPIL compose + FFmpeg NVENC"]
    ED --> OUT(["output.mp4"])

    P -->|"plan.json"| DISK[("Artifacts saved to disk")]
    R -->|"research.json"| DISK
    S -->|"script.json"| DISK
    I -->|"script_final.json"| DISK

    TREND(["Trending Panel"]) -.->|"Pre-fills prompt and skill"| PROMPT

    style P fill:#1e1e28,stroke:#f59e0b,color:#e8eaf0
    style R fill:#1e1e28,stroke:#38bdf8,color:#e8eaf0
    style S fill:#1e1e28,stroke:#38bdf8,color:#e8eaf0
    style QG fill:#1e1e28,stroke:#38bdf8,color:#e8eaf0
    style I fill:#1e1e28,stroke:#38bdf8,color:#e8eaf0
    style TTS_S fill:#1e1e28,stroke:#38bdf8,color:#e8eaf0
    style ED fill:#1e1e28,stroke:#38bdf8,color:#e8eaf0
    style OUT fill:#052e16,stroke:#86efac,color:#86efac
    style DISK fill:#1e1e28,stroke:#6b7280,color:#6b7280
    style TREND fill:#1e1e28,stroke:#f59e0b,color:#e8eaf0
```

---

## 12. Known Problems

| # | Where | What breaks | How bad |
|---|---|---|---|
| 1 | Plan | 8b model misreads complex topics and language | Medium |
| 2 | Plan | User can't review or edit the plan before research starts | Medium |
| 3 | Research | Gap-fill loop only runs once | Low |
| 4 | Research | Long articles truncated at 3000 chars — best facts cut off | Medium |
| 5 | Research | LLM fallback facts have no source URL | Low |
| 6 | Script | Lint rewrite only tried once | Low |
| 7 | Script | Image mode inferred by naive keyword count | Low |
| 8 | Quality gate | Score never shown to user | Medium |
| 9 | Quality gate | Same 5 questions for every skill type | Medium |
| 10 | Images | DuckDuckGo only — no fallback source | Medium |
| 11 | Images | Empty block produces black frames, no hard error | High |
| 12 | Editor | Requires NVIDIA GPU — no CPU path | Medium |
| 13 | Editor | Subtitle preset stuck on minimal — others hidden | Low |
| 14 | TTS | Parallel chunk limit hardcoded, not in profile | Low |
| 15 | LLM | No circuit breaker when primary provider is down | Medium |
| 16 | LLM | Gemini fallback model returns bad JSON frequently | Low |
| 17 | Trending | Jikan and AniList almost never share the same shows | Low |
| 18 | Trending | Only anime is implemented | Medium |
| 19 | Trending | 5s timeout too tight for Jikan | Low |

---

## 13. Roadmap

### Quick wins — under a day each
- [ ] Show quality gate score (N/5) in the UI progress pipeline
- [ ] Add subtitle preset picker to the Agent UI
- [ ] Swap Gemini fallback from `gemma-3-27b-it` to `gemini-2.0-flash`
- [ ] Move TTS parallel chunk limit to `profiles/default.json`
- [ ] Increase Jikan timeout to 10s with one retry

### Medium — 1 to 3 days
- [ ] Plan review step in UI — show topic, queries, and chosen skill before research runs
- [ ] Wikipedia Commons as a second image source for factual topics
- [ ] Per-skill quality gate questions (comedy, dark_secrets, lore_deep_dive each need different criteria)
- [ ] Surface `trending_reason` and `hook_idea` text on trending cards
- [ ] Add a second trending category (gaming via IGDB or music via Billboard)

### Larger — 3 or more days
- [ ] **Agentic RAG for research** — semantic chunking, vector store, hybrid search, parent document retrieval (see Section 3 proposed diagram)
- [ ] **Semantic cache for trending** — embed show summaries, skip the 70b brainstorm call on cache hit (see Section 9 proposed diagram)
- [ ] Circuit breaker for LLM providers — flip order after 3 consecutive failures
- [ ] Hard stop on empty image blocks with a clear user-facing error
- [ ] CPU encode fallback (x264) for machines without NVIDIA GPU
- [ ] ElevenLabs or Kokoro as a fallback TTS engine
