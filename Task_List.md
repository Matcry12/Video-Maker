# Video Maker Implementation Task List

This checklist tracks the development of the 100% free, offline video generation pipeline.

## Phase 1: Environment & Foundation
- [ ] **Project Setup**
    - [ ] Initialize directory structure (as defined in `Detailed_Plan.md`).
    - [ ] Create `requirements.txt` with `vieneu`, `faster-whisper`, `ffmpeg-python`, `pydantic`.
    - [ ] Set up Python Virtual Environment (`venv`).
    - [ ] Verify CUDA/GPU access: `python -c "import torch; print(torch.cuda.is_available())"`.
- [ ] **Configuration & Styles**
    - [ ] Create `profiles/default.json` for role-to-asset mapping.
    - [ ] Create a base `styles.ass` for Advanced Substation Alpha subtitles.

## Phase 2: TTS Module (`src/tts.py`)
- [ ] **Core TTS Logic**
    - [ ] Initialize `VieNeu-TTS` model.
    - [ ] Implement `synthesize(text, lang)` function.
    - [ ] Add audio normalization (ensure consistent volume across clips).
- [ ] **Caching & Alignment**
    - [ ] Implement hash-based caching (skip synthesis if text hasn't changed).
    - [ ] Integrate `faster-whisper` to generate word-level timestamps.
    - [ ] Return a structured object: `(audio_path, duration, timestamps)`.

## Phase 3: Editor Module (`src/editor.py`)
- [ ] **Subtitle Engine**
    - [ ] Create a function to generate `.ass` files from `tts.py` timestamps.
    - [ ] Implement style overrides (colors, font size).
- [ ] **FFmpeg Composition**
    - [ ] Build the background looping filter (`stream_loop`).
    - [ ] Implement vertical video scaling (1080x1920).
    - [ ] **Hardware Acceleration:** Add `h264_nvenc` flags for GTX 1660 SUPER.
- [ ] **Visual Polish**
    - [ ] Implement "Ken Burns" zoom effect for static images.
    - [ ] Implement background music (BGM) mixing with auto-ducking.

## Phase 4: Manager Module (`src/manager.py`)
- [ ] **Orchestration**
    - [ ] Implement Pydantic models for JSON script validation.
    - [ ] Build the "Auto-Asset Mapping" logic (matching `role` to folder/file).
    - [ ] Coordinate the loop: `Script -> TTS -> Alignment -> Editor -> Output`.
- [ ] **Cleanup & Logging**
    - [ ] Implement `tmp/` folder cleanup after successful renders.
    - [ ] Add progress logging (e.g., "Processing Scene 1/4...").

## Phase 5: Testing & Optimization
- [ ] **Functional Testing**
    - [ ] Run full render with `Script_Template.json`.
    - [ ] Verify Vietnamese tone accuracy in subtitles.
- [ ] **Hardware Benchmarking**
    - [ ] Monitor VRAM usage during render (ensure it stays under 6GB).
    - [ ] Measure total time for a 60-second video (target: < 2 min).

## Phase 6: Assets & Content
- [ ] **Asset Gathering**
    - [ ] Populate `assets/videos/intro` and `assets/videos/tips`.
    - [ ] Download 5-10 royalty-free Lo-Fi background tracks.
- [ ] **Template Expansion**
    - [ ] Create additional profiles (e.g., `motivation.json`, `fact_channel.json`).

## Phase 7: Subtitle Quality Upgrade
- [x] **Do First: Segmentation & Timing Engine**
    - [x] Replace fixed character chunking with phrase-level segmentation.
    - [x] Enforce 1-2 subtitle lines per caption event.
    - [x] Add timing guardrails (minimum/maximum duration per caption).
    - [x] Apply reading-speed rule (~12-16 characters/second for Vietnamese).
    - [x] Smooth adjacent caption gaps/overlaps for natural flow.
- [x] **Do First: Text Normalization**
    - [x] Normalize spaces and punctuation before subtitle generation.
    - [x] Prevent awkward word splits and isolated filler words.
    - [x] Preserve Vietnamese diacritics and avoid long all-caps captions.
    - [x] Align caption boundaries with speech pauses where possible.
- [x] **Do Next: Visual Styling Improvements**
    - [x] Update default subtitle style for mobile readability (bold + outline + shadow).
    - [x] Reserve subtitle safe area to avoid overlap with Shorts/Reels UI.
    - [x] Add optional keyword highlight with a single accent color.
    - [x] Add optional progressive reveal/karaoke emphasis mode.
- [x] **Do Later: Preset System**
    - [x] Define subtitle preset: `minimal`.
    - [x] Define subtitle preset: `energetic`.
    - [x] Define subtitle preset: `cinematic`.
    - [x] Add config switch to choose subtitle preset per script/profile.
- [ ] **Final Step: Validation & A/B Testing**
    - [ ] Render the same script with all subtitle presets for comparison.
    - [ ] Review readability on mobile (1080x1920) before publishing.
    - [ ] Track watch metrics (average watch time, early drop-off, readability feedback).
    - [ ] Finalize one default subtitle preset after test results.
