# Video Maker — Full Pipeline Flow

```mermaid
flowchart TD
    USER(["👤 User Prompt\n'Sung Jinwoo Shadow Monarch powers'"])

    subgraph PLAN ["① PLAN STAGE — plan_agent.py"]
        P1["Parse user prompt\nExtract topic, category, angles"]
        P2["Generate 7 search queries\n(angle-specific BM25 queries)"]
        P3["Output: AgentPlan\n• topic\n• must_cover angles\n• entity_aliases\n• search_queries"]
        P1 --> P2 --> P3
    end

    subgraph RESEARCH ["② RESEARCH STAGE — research_agent.py"]
        direction TB

        R1["DuckDuckGo + Wikipedia search\n7 queries → ~70-80 URLs"]

        subgraph CRAWL ["Crawl Phase"]
            RC1["BM25 score each URL\nagainst must_cover angles"]
            RC2["Fetch top 8 pages per angle\n(skip bot-blocked sites)"]
            RC3["Semantic chunking\n• Split at sentence boundaries\n• 800 char target, overlap 100"]
            RC1 --> RC2 --> RC3
        end

        subgraph RAG ["RAG Store — rag_store.py (ChromaDB)"]
            RG1["Store new chunks\nwith metadata + embeddings"]
            RG2["Hybrid Retrieve\n① Vector search (MiniLM-L12)\n② BM25 keyword search\n③ RRF merge (tier-weighted)\n④ Cross-encoder rerank\n   (ms-marco-MiniLM-L-6-v2)"]
            RG1 --> RG2
        end

        subgraph EXTRACT ["Extraction Phase"]
            RE1["Batch all top-20 chunks\ninto ONE LLM call"]
            RE2["Pass 1: angle-specific\n(skill extraction_hint)"]
            RE3["Pass 2: broad scan\n(if < 6 facts found)"]
            RE2 --> RE3
        end

        subgraph VERIFY ["Grounding & Quality"]
            RV1["Fuzzy-match verbatim_evidence\nagainst source text"]
            RV2["Jaccard dedup\n(threshold 0.5)"]
            RV3["Quality floor check\n• ≥ 6 grounded facts?\n• ≥ 2 source domains?\n• Every must_cover angle covered?"]
            RV1 --> RV2 --> RV3
        end

        REFLECT["Reflect loop (max 2x)\nGap queries → re-crawl uncovered angles"]

        R1 --> CRAWL --> RAG
        RAG --> EXTRACT
        RE1 --> RE2
        EXTRACT --> VERIFY
        RV3 -->|"Floor not met"| REFLECT
        REFLECT -->|"New URLs"| CRAWL
    end

    subgraph SCRIPT ["③ SCRIPT STAGE — script_agent.py + fact_script_writer.py"]
        S1["Select skill via BM25\n(lore_deep_dive, comparison, top_list…)"]
        S2["Build system prompt\n• skill hook_rule (with topic substituted)\n• banned slop phrases\n• must_cover angles\n• citation requirement"]
        S3["LLM: write single-block narration\n≥ 200 words, TTS-friendly"]
        S4["Lint check\n• hook specificity\n• sentence length\n• duplicate ending\n• word count"]
        S5{"Lint score ≥ 80?"}
        S6["Rewrite with feedback\n(up to 3 attempts)"]
        S7["Phrase windows\nsplit_into_windows()\n~40 words / window\nper-window image keywords"]
        S1 --> S2 --> S3 --> S4 --> S5
        S5 -->|"No"| S6 --> S4
        S5 -->|"Yes"| S7
    end

    subgraph GATE ["④ QUALITY GATE — quality_gate.py"]
        G1["Deterministic checks (0-60 pts)\n• Grounding (citation rate)\n• Contamination (cross-franchise)\n• Hook specificity\n• Loop-back (no duplicate ending)\n• Sentence length\n• Must-cover coverage"]
        G2["LLM judge (0-50 pts)\n5 questions:\n• Hook shocking enough?\n• Lore vs recap?\n• Surprise moment?\n• Seamless loop?\n• Escalating tension?"]
        G3{"Combined ≥ threshold?"}
        G4["Rewrite with gate feedback\n(1 attempt)"]
        G5["Re-attach phrase windows\n(fix for rewrite dropping them)"]
        G1 --> G2 --> G3
        G3 -->|"No"| G4 --> G5
        G3 -->|"Yes"| G5
    end

    subgraph IMAGES ["⑤ IMAGE STAGE — image_agent.py + pipeline.py"]
        I1["Per-window keyword synthesis\n(stopword-filtered proper nouns)"]
        I2["Search images per window\nWikimedia + web sources"]
        I3["Keywords-first search\nper-window → topic fallback"]
        I4["Download & cache images\n(dedup by URL)"]
        I1 --> I2 --> I3 --> I4
    end

    subgraph EDITOR ["⑥ EDITOR STAGE — editor_agent.py + manager.py + tts.py"]
        E1["TTS synthesis\nEdge-TTS, rate=-18%\nStrip citation markers [F001]"]
        E2["Word-level alignment\n(subtitle timing)"]
        E3["ASS subtitle generation"]
        E4["PIL pre-compose frames\nimage per phrase window"]
        E5["FFmpeg rawvideo pipe\n(27x faster than overlay chain)"]
        E6["Mix BGM\n(mood-based, 0.15 volume)"]
        E7["Export MP4"]
        E1 --> E2 --> E3
        E4 --> E5
        E3 --> E5
        E5 --> E6 --> E7
    end

    USER --> PLAN
    PLAN --> RESEARCH
    RESEARCH -->|"6-12 grounded facts\n+ angle_served tags"| SCRIPT
    SCRIPT -->|"single-block script\n+ 8-12 windows"| GATE
    GATE -->|"approved script"| IMAGES
    IMAGES -->|"script + image_map"| EDITOR
    EDITOR --> OUTPUT(["🎬 output/video.mp4"])

    %% Model annotations
    classDef groqFast fill:#f0f4ff,stroke:#4a6fa5
    classDef groqBig fill:#e8f4fd,stroke:#2980b9
    classDef gemini fill:#e8f8e8,stroke:#27ae60
    classDef local fill:#fff3e0,stroke:#e67e22

    M_PLAN["🤖 plan\nGroq: llama-3.3-70b\nGemini fallback: gemma-3-27b-it"]:::groqBig
    M_CRAWL["🤖 crawl / interest_rank\nbank_extract / research_eval\nGroq only: llama-3.1-8b-instant"]:::groqFast
    M_EXTRACT["🤖 research_extract\nGemini primary: gemma-4-31b-it\nGroq fallback: llama-3.3-70b"]:::gemini
    M_SCRIPT["🤖 script\nGemini primary: gemma-4-31b-it\nGroq fallback: llama-3.3-70b"]:::gemini
    M_GATE["🤖 quality_gate\nGemini primary: gemma-4-31b-it\nGroq fallback: llama-3.3-70b"]:::gemini
    M_EMBED["🔍 embeddings (local)\nMiniLM-L12 (RAG store)\nms-marco-MiniLM-L-6 (reranker)"]:::local

    M_PLAN -.->|used by| PLAN
    M_CRAWL -.->|used by| CRAWL
    M_EXTRACT -.->|used by| EXTRACT
    M_SCRIPT -.->|used by| SCRIPT
    M_GATE -.->|used by| GATE
    M_EMBED -.->|used by| RAG
```

---

## RAG Retrieval Detail

```mermaid
flowchart LR
    Q["Queries\n(up to 6)"]

    subgraph VECTOR ["Vector Leg"]
        V1["Embed query\nMiniLM-L12"]
        V2["ChromaDB ANN search\ntop 36 candidates"]
        V1 --> V2
    end

    subgraph BM25L ["BM25 Leg"]
        B1["Tokenize all stored docs"]
        B2["BM25Okapi score\nagainst query tokens"]
        B3["Top 36 by keyword score"]
        B1 --> B2 --> B3
    end

    subgraph FUSION ["RRF Merge"]
        F1["Reciprocal Rank Fusion\nscore = tier_weight / (rank + 60)\nTier 1 → 1.5×, Tier 4 → 0.25×"]
        F2["Top 24 candidates"]
        F1 --> F2
    end

    subgraph RERANK ["Cross-Encoder Rerank"]
        CR1["ms-marco-MiniLM-L-6-v2\nscore (query, chunk) pairs"]
        CR2["Re-sort by ce_score"]
        CR3["Return top 12"]
        CR1 --> CR2 --> CR3
    end

    Q --> VECTOR
    Q --> BM25L
    VECTOR --> FUSION
    BM25L --> FUSION
    FUSION --> RERANK
```

---

## Known Issues & Future Work

| Area | Issue | Fix |
|------|-------|-----|
| Rate limits | `crawl`, `interest_rank`, `bank_extract`, `research_eval` all hit Groq 8B rate limit | Move to local Ollama E2B/E4B |
| Image validation | No check if downloaded image is actually relevant | Use Ollama E4B vision to score images |
| RAG duplicate IDs | Same URL processed across runs causes `upsert` warning | Deduplicate by URL hash before upsert |
| Research facts quality | Small 8B model for extraction misses subtle facts | Use gemma-4-31b-it for extraction (partially done) |
| Script ending | Loop-back sometimes copies hook verbatim | Duplicate ending penalty added to gate |
