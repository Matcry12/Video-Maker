# TTS Integration Plan — Kokoro for English, Edge-TTS for Vietnamese

## Decision (validated in lab)

| Use case | Engine | Voice | Reason |
|---|---|---|---|
| English Shorts (US monetization, current) | **Kokoro-ONNX** | `am_liam` | Better natural prosody than Edge-TTS at acceptable speed |
| Vietnamese (future, secondary) | **Edge-TTS** (unchanged) | `vi-VN-NamMinhNeural` | Kokoro is English-only |
| Storytelling (future, longer-form) | **Chatterbox standard** | default + `exaggeration=0.7` | Emotion control, deferred |

Lab evidence (`lab/tts/full_pipeline_chunked/am_liam/`):
- 100% Whisper alignment coverage on 211-word script
- 75.9s audio, 0.30s tail drift, 9 gaps >500ms
- 23.8s generation time (3.2x realtime, CPU)

## Architecture

Add a **dispatcher pattern** in `src/tts.py` — `synthesize()` picks the engine based on language.

```
TTSEngine.synthesize(text, output_path, voice=None, language=None, ...)
  ├─ language == "en" / voice starts with af_/am_  → KokoroBackend
  ├─ default                                        → EdgeBackend (existing)
  └─ both return identical dict: {audio_path, duration, words, ...}
```

The existing `TTSEngine` class becomes a thin facade. Both backends share the same word-timestamp post-processing (`_postprocess_word_timestamps`) and caching helpers.

**Critical: Kokoro path forces `alignment_mode="forced"`** because Kokoro emits no native word boundaries. Whisper alignment is already wired up at line 484 of current `src/tts.py`.

## Chunking (mandatory for Kokoro)

Lab proved one-shot Kokoro on >1000 chars degrades (Fenrir went robotic after 6s). Per-sentence chunking restored quality.

Use the existing `_split_into_chunks` regex from `src/tts.py:73-78`. Same sentence-boundary logic, no rework needed. Default `MAX_CHUNK_CHARS=200` for Kokoro (vs current 500 for Edge-TTS — Kokoro is more sensitive to length).

Chunks generate sequentially (Kokoro is GIL-bound on CPU; no benefit from `asyncio.gather`). Each chunk's PCM is concatenated with `np.concatenate` — same merge code as `_merge_chunk_audio`.

## File-by-file changes

| File | Change | Lines (est) |
|---|---|---|
| `src/tts.py` | Add `KokoroBackend` class with `synthesize()` returning the existing dict shape; refactor `TTSEngine.synthesize()` to dispatch | +200, -30 |
| `src/tts.py` | Helper to download `kokoro-v1.0.onnx` + `voices-v1.0.bin` into `assets/models/kokoro/` on first run | +30 |
| `profiles/default.json` | Add `tts.engine_by_language: {"en": "kokoro", "vi": "edge"}` and `tts.kokoro: {voice, max_chunk_chars, providers}` | +8 |
| `requirements.txt` | Add `kokoro-onnx>=0.5.0`, `onnxruntime>=1.20` | +2 |
| `src/agent_config.py` | Surface new tts section via `load_agent_settings()` (probably already does) | 0–5 |

**Files NOT touched:**
- `src/whisper_align.py` — already does forced alignment, no changes
- `src/agent/editor_agent.py` — just consumes the `words` list, engine-agnostic
- `src/editor.py` — consumes audio path + words, engine-agnostic
- All subtitle / image / render code — downstream of the same dict shape

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| `kokoro-onnx` install conflicts with main `.venv` (torch 2.11) | Pure-ONNX has zero torch dependency. Verified compatible in `.venv-tts`. Smoke test before merging. |
| Whisper alignment adds ~7s per 60s script | Already accepted cost for `alignment_mode="forced"`. Cache aligned words alongside audio. |
| Model file is 325MB (ONNX) + 28MB (voices) | Download lazily on first run. Add `assets/models/kokoro/` to `.gitignore`. |
| Voice `am_liam` sounds different in production from lab (model nondeterminism) | Kokoro-ONNX is **deterministic** — same input always produces identical output. Verified. |
| GPU CUDA-12 mismatch (lab fell back to CPU) | Acceptable — CPU at 3.2x realtime is fine for batch. GPU fix deferred. |
| Cache key compatibility with existing Edge-TTS cache | Use distinct cache key prefix: `kokoro_<hash>` vs current `<hash>`. Safe. |

## Phased rollout

**Phase 1 — Backend integration (this PR, ~3 hr):**
1. Smoke test: install kokoro-onnx in main `.venv`, verify torch unaffected
2. Add `KokoroBackend` class
3. Add dispatcher in `TTSEngine.synthesize()`
4. Add profile config
5. Run end-to-end pipeline on the JoJo script via `src/web.py` and verify final MP4
6. Compare A/B: Edge-TTS MP4 vs Kokoro MP4

**Phase 2 — Tuning (follow-up, deferred):**
- GPU support (install nvidia-cublas-cu12, set `providers=["CUDA..."]`)
- Per-chunk crossfade if seam clicks audible
- Voice fallback if `am_liam` model file missing

**Phase 3 — Storytelling track (future, separate PR):**
- Chatterbox standard backend with exaggeration param
- Profile route `tts.engine_by_skill: {"long_form": "chatterbox"}`
- Probably stays in `.venv-tts` until the integration is needed

## Decisions locked

1. **Voice config = hybrid (profile default + skill override).**
   - `profiles/default.json` sets `tts.kokoro.voice = "am_liam"` as default.
   - Each `skills/*.json` may add `tts_voice: "..."` to override per content type.
   - Resolution order: `skill.tts_voice` → `profile.tts.kokoro.voice` → builtin default `am_liam`.
   - Sets up storytelling skill (Chatterbox) for future without another refactor.

2. **Edge-TTS fallback ON.** If `kokoro-onnx` fails to import OR model file is missing OR synthesis raises, fall back to `en-US-GuyNeural` (Edge-TTS) and log a warning. Pipeline never crashes on TTS errors.

3. **Cache key includes voice.** Already does — voice param hashed into cache key. Safe to swap voice without manual cache invalidation.
