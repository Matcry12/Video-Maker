# Plan Feedback & Feasibility Report

## 1. Hardware Alignment Analysis
- **CPU (i7-10700):** Excellent for this workload. With 16 threads, you can handle the orchestration and CPU-based TTS synthesis (if the model doesn't support CUDA) without significant bottlenecks.
- **RAM (16GB):** Sufficient for the proposed stack. However, running `faster-whisper` (Large-v3) and `VieNeu-TTS` simultaneously might push usage to ~10-12GB. Avoid heavy background apps during bulk processing.
- **GPU (GTX 1660 SUPER):** This is the "secret weapon" for your speed. 
    - **VRAM (6GB):** Enough for `faster-whisper` (using `float16` or `int8` quantization) and NVENC encoding.
    - **NVENC:** Will allow for near real-time video rendering, which is much faster than CPU encoding.

## 2. Potential Bottlenecks
- **TTS Synthesis Speed:** Local TTS is generally slower than cloud APIs. On your i7, synthesizing 1 minute of audio might take 15-30 seconds. Batching scripts will be essential for productivity.
- **Vietnamese Subtitle Accuracy:** While `faster-whisper` is world-class, Vietnamese tone marks can occasionally be dropped in noisy backgrounds. 
    - *Correction:* Since you are generating the audio from text, you should use the **original text** for subtitles rather than transcribing the audio, OR use a forced-aligner like `aeneas` to get timestamps for the existing text. This guarantees 100% text accuracy.

## 3. Storage Considerations
- Local models (VieNeu, Whisper) will take ~3-5 GB.
- High-quality 1080p stock footage can fill up 16GB of RAM quickly if not handled correctly. Ensure you are using FFmpeg's stream-looping instead of loading full files into memory.

## 4. Strategic Recommendations
1. **Prefer Alignment over Transcription:** For subtitles, use the JSON script text and align it to the audio timestamps. This is more reliable than "hearing" the TTS.
2. **Quantization:** Run Whisper in `int8` or `float16` mode to save VRAM on your 1660 SUPER.
3. **NVENC Hardware Acceleration:** Explicitly use `-c:v h264_nvenc` in your FFmpeg commands. This will reduce render times by up to 5x compared to the default `libx264`.
4. **Font Selection:** For Vietnamese, ensure your chosen font in `styles.ass` supports all Unicode characters (e.g., *Be Vietnam Pro*, *Roboto*, or *Noto Sans*).

## 5. Architectural Benefits (Modular Approach)
- **Memory Management:** By splitting `tts.py`, `editor.py`, and `manager.py`, the system can release RAM/VRAM between stages. For example, `faster-whisper` models can be unloaded before the `editor.py` starts the heavy FFmpeg NVENC render.
- **Maintainability:** Modularizing the TTS logic makes it trivial to swap `VieNeu-TTS` for a different engine in the future without touching the video rendering logic.
- **Error Isolation:** If a TTS synthesis fails, the `manager.py` can retry that specific chunk without restarting the entire video project.

## 6. Overall Feasibility Score: 9.5/10
The transition to a modular architecture increases the score from 9 to 9.5. Your hardware is now even better positioned to handle this stack due to the improved resource lifecycle management inherent in a decoupled design.
