# Detailed Video Maker Implementation Plan (March 2026)

This document outlines the technical implementation for a 100% free, offline-capable video generation pipeline, optimized for the existing hardware (i7-10700 + GTX 1660 SUPER).

## 1. System Architecture
The system follows a modular, decoupled architecture to ensure reusability and ease of updates. It is designed for a **Content-First Workflow**, where the user provides text/roles and the system automates asset selection.

### Core Modules:
- **`tts.py` (Voice Engine):** Handles Text-to-Speech synthesis (VieNeu-TTS) and audio post-processing.
- **`editor.py` (Visual Engine):** Handles FFmpeg composition, subtitle styling (.ass), and hardware-accelerated rendering.
- **`manager.py` (Orchestrator):** The brain of the operation. It maps content `roles` to visual assets and coordinates the TTS and Editor modules.

## 2. Phase 1: Environment Setup
### Python Environment
- Create a virtual environment: `python -m venv venv`
- Essential libraries:
  ```bash
  pip install vieneu ffmpeg-python faster-whisper pydantic torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- **Note:** Installing the CUDA-enabled version of PyTorch is critical to utilize the GTX 1660 SUPER.

### Hardware-Specific Configuration
- **GPU (GTX 1660 SUPER):** Use for `faster-whisper` (large-v3 model) and FFmpeg `h264_nvenc` encoding.
- **CPU (i7-10700):** 16 threads available for parallelizing TTS synthesis if multiple videos are being processed in batch.

## 3. Phase 2: Modular Module Implementation

### A. TTS Module (`tts.py`)
- **Responsibility:** Convert text to high-quality audio.
- **Features:** 
    - Mixed Vietnamese/English support (via `sea-g2p`).
    - Audio normalization and noise reduction.
    - Caching mechanism: Checks if the exact text/voice combo was already generated.
- **Logic:** Wraps `vieneu` and `faster-whisper` for alignment.

### B. Editor Module (`editor.py`)
- **Responsibility:** All visual and final assembly tasks.
- **Features:**
    - `.ass` Subtitle generation with custom fonts (Be Vietnam Pro).
    - Dynamic video effects (Ken Burns zoom, overlays).
    - **NVENC Acceleration:** Optimized for GTX 1660 SUPER.
- **Logic:** Builds complex FFmpeg filter chains based on scene metadata.

### C. Manager Module (`manager.py`)
- **Responsibility:** Orchestrate the "Content-to-Video" pipeline.
- **Features (Auto-Asset Mapping):**
    - **Profile Loading:** Reads `profiles/default.json` to map `roles` to asset directories.
        - Example: `role: "intro"` -> `assets/videos/intro_loop.mp4`
        - Example: `role: "tip"` -> Random clip from `assets/videos/finance_stock/`
    - **Validation:** Ensures JSON scripts match the simplified `language` + `blocks` structure.
    - **Coordination:** Feeds text to `tts.py`, then takes audio metadata + mapped assets and feeds them to `editor.py`.

## 4. Phase 3: Workflow Automation
1. **Script Input:** User provides a high-level JSON (e.g., `3_tips_save_money.json`).
2. **Auto-Mapping:** Manager scans the script and assigns backgrounds/music based on the `language` and `role`.
3. **Execution Loop:**
   - Manager sends text to `tts.py`.
   - `tts.py` returns audio path + precise timestamp data.
   - Manager sends mapped assets + audio + timestamps to `editor.py`.
   - `editor.py` performs the final hardware-accelerated render (NVENC).

## 5. Directory Structure
```text
/video-maker
├── assets/             # Subfolders for videos, images, BGM
│   ├── videos/         # Stock footage (intro, tips, outro)
│   └── audio/          # Background music tracks
├── json_scripts/       # Your simplified content scripts
├── profiles/           # Asset mapping & global configs (fonts, colors)
├── output/             # Final generated MP4 files
├── tmp/                # Audio chunks, intermediate logs
├── src/                # Source code
│   ├── tts.py          # TTS & Alignment logic
│   ├── editor.py       # FFmpeg & Visual logic
│   └── manager.py      # Orchestrator (Auto-mapping logic)
├── styles.ass          # Global subtitle style template
└── requirements.txt    # Dependency list
```

## 6. Success Metrics
- **Throughput:** Aim for < 2 minutes render time for a 60s 1080p Short.
- **Efficiency:** Writing a new video script should take < 5 minutes of text work.
- **Quality:** Word-accurate synchronization for Vietnamese tones.

## 7. Subtitle Improvement Plan (Priority Update)
Current subtitle quality is below target for short-form retention. The next subtitle pass should focus on readability, pacing, and visual clarity instead of simple character-based splitting.

### A. Segmentation & Timing Rules
- Segment subtitles by meaning (phrase-level chunks), not fixed character length.
- Keep captions to 1-2 lines per event; avoid 3-line blocks on vertical video.
- Target caption duration around 1.0-2.5 seconds.
- Enforce reading speed limit: ~12-16 characters/second for Vietnamese.
- Add timing guardrails:
  - Minimum subtitle duration to avoid fast flashing.
  - Maximum subtitle duration to prevent long static captions.
  - Small overlap/gap smoothing between adjacent events.

### B. Text Cleanup Rules Before Render
- Normalize spacing and punctuation before subtitle generation.
- Prevent awkward token splits (e.g., punctuation or short filler words isolated on a line).
- Preserve Vietnamese diacritics and avoid full-uppercase long captions.
- Prefer stable phrase boundaries aligned to pauses in speech.

### C. Visual Style Rules (.ass)
- Use a strong readable font with high contrast against mixed backgrounds.
- Keep bold text + clear outline + light shadow as default.
- Reserve safe subtitle area to avoid overlap with Shorts/Reels UI.
- Highlight only key words with one accent color (no multi-color overload).
- Consider progressive reveal/karaoke emphasis for key phrases, not full-line flicker.

### D. Presets & Evaluation
- Define 2-3 subtitle presets for quick testing:
  - Minimal: clean and neutral.
  - Energetic: stronger emphasis on keywords.
  - Cinematic: slower pacing and larger margins.
- A/B test presets on real videos and monitor:
  - Average watch duration.
  - Early drop-off (first 3-5 seconds).
  - Subjective readability feedback.

### E. Done Criteria
- Subtitles are easy to read at normal mobile viewing speed.
- Caption transitions feel natural with speech rhythm.
- No obvious awkward breaks or UI overlap in 1080x1920 exports.
