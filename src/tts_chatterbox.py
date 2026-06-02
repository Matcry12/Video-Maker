"""Chatterbox TTS backend — voice cloned narration for Shorts and long-form.

Lab-validated configuration (see LONG_FORM_TTS_FINDINGS.md, 2026-04-30):
  - Standard ChatterboxTTS (not Turbo)
  - audio_prompt_path = pre-built reference clip from assets/voices/chatterbox/
  - exaggeration = 0.7, cfg_weight = 0.1
  - paragraph-aware chunker with kind-keyed gaps (350ms paragraph,
    200ms block, 50ms crossfade for sub-chunks)

Public API mirrors _KokoroBackend in src/tts.py:
  backend = _ChatterboxBackend.get()
  duration_sec, sample_rate = backend.synthesize_to_wav(text, voice, out_path)

`voice` is the short name without the `cb:` prefix — e.g. "heart" or "fenrir".
The full reference clip lives at assets/voices/chatterbox/<voice>.wav.

Dtype monkey-patches for chatterbox 0.1.7 are applied at module import.
Without them the s3tokenizer + voice_encoder paths crash on float32/float64
mismatches when given a reference clip.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torchaudio as ta

from .agent_config import load_agent_settings

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
REFS_DIR = ROOT / "assets" / "voices" / "chatterbox"

VOICE_PREFIX = "cb:"

_cfg = (load_agent_settings().get("tts", {}) or {}).get("chatterbox", {}) or {}
DEFAULT_VOICE = str(_cfg.get("voice", "fenrir"))
DEFAULT_EXAG = float(_cfg.get("exaggeration", 0.7))
DEFAULT_CFG = float(_cfg.get("cfg_weight", 0.3))
DEFAULT_DEVICE = str(_cfg.get("device", "auto")).lower()


def is_chatterbox_voice(voice: Optional[str]) -> bool:
    """True if voice id is in the cb: namespace."""
    return bool(voice) and voice.startswith(VOICE_PREFIX)


def strip_prefix(voice: str) -> str:
    return voice[len(VOICE_PREFIX):] if voice.startswith(VOICE_PREFIX) else voice


def reference_path(voice_short: str) -> Path:
    """Resolve short voice name to its reference clip path.

    Raises FileNotFoundError if the clip is missing — caller should fall back
    or surface the error.
    """
    p = REFS_DIR / f"{voice_short}.wav"
    if not p.exists():
        raise FileNotFoundError(f"Chatterbox reference clip missing: {p}")
    return p


# ---------------------------------------------------------------------------
# Smart chunker — block > paragraph > sentence > clause priority.
# Same logic as lab/longvideo/agent.py, ported here.
# ---------------------------------------------------------------------------
PARA_BUDGET = 500
HARD_MAX = 600
ABBREVS = ("Mr.", "Mrs.", "Ms.", "Dr.", "St.", "vs.", "etc.", "e.g.", "i.e.")
TRANSITION_WORDS = (
    "However", "Suddenly", "Meanwhile", "Now", "Later", "Finally",
    "Eventually", "Then", "But", "Years later", "The next morning",
    "After", "Before",
)
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\["\'])')


@dataclass
class Chunk:
    text: str
    kind: str  # "block" | "paragraph" | "sub"


def _split_sentences_smart(text: str) -> list[str]:
    raw = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    out: list[str] = []
    for s in raw:
        if out and any(out[-1].endswith(a) for a in ABBREVS):
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return out


def _pack_sentences(sents: list[str], budget: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sents:
        if cur and cur_len + len(s) + 1 > budget:
            chunks.append(" ".join(cur))
            cur, cur_len = [s], len(s)
        else:
            cur.append(s)
            cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _split_oversize_paragraph(para: str, budget: int) -> list[str]:
    sents = _split_sentences_smart(para)
    groups: list[list[str]] = [[]]
    for s in sents:
        first_word = s.split()[0] if s.split() else ""
        if groups[-1] and (
            first_word in TRANSITION_WORDS
            or any(s.startswith(t + " ") for t in TRANSITION_WORDS if " " in t)
        ):
            groups.append([])
        groups[-1].append(s)
    out: list[str] = []
    for g in groups:
        if not g:
            continue
        joined = " ".join(g)
        if len(joined) <= budget:
            out.append(joined)
        else:
            out.extend(_pack_sentences(g, budget))
    return out


def chunk_text(passage: str, *, blocks: list[dict] | None = None) -> list[Chunk]:
    if blocks:
        return [Chunk(text=b["text"].strip(), kind="block")
                for b in blocks if b.get("text", "").strip()]
    paras = [p.strip() for p in re.split(r'\n\s*\n+', passage.strip()) if p.strip()]
    out: list[Chunk] = []
    for para in paras:
        para = re.sub(r"\s+", " ", para)
        if len(para) <= PARA_BUDGET:
            out.append(Chunk(text=para, kind="paragraph"))
        else:
            subs = _split_oversize_paragraph(para, PARA_BUDGET)
            for i, s in enumerate(subs):
                kind = "paragraph" if i == 0 else "sub"
                out.append(Chunk(text=s, kind=kind))
    if not out and passage.strip():
        out.append(Chunk(text=re.sub(r"\s+", " ", passage.strip()), kind="paragraph"))
    _validate_chunks(out)
    return out


def _validate_chunks(chunks: list[Chunk]) -> None:
    for c in chunks:
        t = c.text
        if not t:
            raise ValueError("empty chunk")
        if t[0] not in '"\'[' and not t[0].isupper():
            raise ValueError(f"chunk starts mid-sentence: {t[:80]!r}")
        if t.rstrip()[-1] not in '.!?"\'':
            raise ValueError(f"chunk ends mid-sentence: {t[-80:]!r}")
        if t.count('"') % 2 != 0:
            raise ValueError(f"unbalanced quotes: {t[:80]!r}")
        if len(t) > HARD_MAX:
            raise ValueError(f"chunk too long ({len(t)} > {HARD_MAX}): {t[:80]!r}")


# ---------------------------------------------------------------------------
# Audio concat with kind-keyed gaps
# ---------------------------------------------------------------------------
def _trim_leading_silence(wav: torch.Tensor, sr: int,
                           threshold: float = 0.005,
                           max_trim_sec: float = 0.08) -> torch.Tensor:
    """Remove the ~30ms near-silent preamble Chatterbox generates before speech onset.

    Without this, the fade-in ramp reaches full scale right as the first consonant
    transient hits, leaving that attack fully exposed and perceived as a pop.
    Trims up to max_trim_sec of leading near-silence; stops at first active frame.
    """
    w = wav if wav.dim() == 1 else wav[0]
    win = max(1, int(sr * 0.002))   # 2ms windows
    max_trim = int(sr * max_trim_sec)
    cut = 0
    for start in range(0, min(max_trim, w.shape[0] - win), win):
        if w[start:start + win].abs().max().item() > threshold:
            cut = start
            break
    return wav[..., cut:] if cut > 0 else wav


def _gap_samples(prev_kind: str, cur_kind: str, sr: int) -> int:
    if cur_kind == "block":
        return int(sr * 0.20)
    if cur_kind == "paragraph":
        return int(sr * 0.35)
    return 0  # sub: crossfade only


def _concat_with_gaps(waves: list[torch.Tensor], kinds: list[str], sr: int,
                      fade_sec: float = 0.05) -> torch.Tensor:
    if not waves:
        return torch.zeros(1, 0)

    def _norm(w: torch.Tensor) -> torch.Tensor:
        if w.dim() == 2 and w.shape[0] > 1:
            w = w.mean(dim=0, keepdim=True)
        elif w.dim() == 1:
            w = w.unsqueeze(0)
        return w

    waves = [_norm(w) for w in waves]
    out = waves[0]
    fade_len_full = max(1, int(sr * fade_sec))
    for i in range(1, len(waves)):
        nxt = waves[i]
        gap = _gap_samples(kinds[i - 1], kinds[i], sr)
        if gap > 0:
            silence = torch.zeros(out.shape[0], gap)
            out = torch.cat([out, silence, nxt], dim=-1)
        else:
            f = min(fade_len_full, out.shape[-1], nxt.shape[-1])
            if f <= 1:
                out = torch.cat([out, nxt], dim=-1)
                continue
            ramp = torch.linspace(0.0, 1.0, f).unsqueeze(0)
            head = out[:, :-f]
            tail_a = out[:, -f:]
            head_b = nxt[:, :f]
            rest_b = nxt[:, f:]
            mixed = tail_a * (1.0 - ramp) + head_b * ramp
            out = torch.cat([head, mixed, rest_b], dim=-1)
    return out


# ---------------------------------------------------------------------------
# Dtype patches for chatterbox 0.1.7 (mandatory before any reference call)
# ---------------------------------------------------------------------------
_PATCHED = False
_PATCH_LOCK = threading.Lock()


def _patch_chatterbox_dtype() -> None:
    global _PATCHED
    with _PATCH_LOCK:
        if _PATCHED:
            return
        from chatterbox.models.s3tokenizer import s3tokenizer as s3t
        from chatterbox.models.voice_encoder import voice_encoder as ve

        _orig_mel = s3t.S3Tokenizer.log_mel_spectrogram

        def _mel_patched(self, audio, padding: int = 0):
            if not torch.is_tensor(audio):
                audio = torch.from_numpy(audio)
            audio = audio.to(dtype=torch.float32, device=self.device)
            return _orig_mel(self, audio, padding=padding)

        s3t.S3Tokenizer.log_mel_spectrogram = _mel_patched

        _orig_inf = ve.VoiceEncoder.inference

        def _inf_patched(self, mels, mel_lens, *args, **kwargs):
            if torch.is_tensor(mels):
                mels = mels.to(dtype=torch.float32)
            return _orig_inf(self, mels, mel_lens, *args, **kwargs)

        ve.VoiceEncoder.inference = _inf_patched
        _PATCHED = True
        logger.debug("Chatterbox dtype patches applied.")


# ---------------------------------------------------------------------------
# Backend singleton
# ---------------------------------------------------------------------------
class _ChatterboxBackend:
    """Lazy-loaded ChatterboxTTS singleton.

    Single GPU model + single inference lock. Concurrent synthesize() calls
    serialize on the lock so we don't OOM the card.
    """

    _instance: "Optional[_ChatterboxBackend]" = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "_ChatterboxBackend":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def __init__(self):
        _patch_chatterbox_dtype()
        device = self._resolve_device()
        from chatterbox.tts import ChatterboxTTS
        logger.info("Loading ChatterboxTTS (standard) on %s ...", device)
        t0 = time.time()
        self.model = ChatterboxTTS.from_pretrained(device=device)
        load_sec = time.time() - t0
        self.device = device
        self.sr = int(self.model.sr)
        self.exag = DEFAULT_EXAG
        self.cfg = DEFAULT_CFG
        self._gen_lock = threading.Lock()
        logger.info(
            "Chatterbox ready (device=%s, sr=%d, exag=%.2f, cfg=%.2f, load=%.1fs)",
            device, self.sr, self.exag, self.cfg, load_sec,
        )

    @staticmethod
    def _resolve_device() -> str:
        if DEFAULT_DEVICE == "cpu":
            return "cpu"
        if DEFAULT_DEVICE == "cuda":
            if not torch.cuda.is_available():
                logger.warning("Chatterbox device=cuda requested but CUDA unavailable; using CPU.")
                return "cpu"
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def synthesize_to_wav(self, text: str, voice: str, output_path: Path) -> tuple[float, int]:
        """Render `text` with reference voice `voice` ("heart"/"fenrir") to wav.

        Returns (duration_sec, sample_rate). Chunks internally; concatenates
        with kind-keyed silence + crossfades.
        """
        ref_path = reference_path(voice)
        chunks = chunk_text(text)
        kwargs = {
            "audio_prompt_path": str(ref_path),
            "exaggeration": self.exag,
            "cfg_weight": self.cfg,
        }

        waves: list[torch.Tensor] = []
        with self._gen_lock:
            for i, c in enumerate(chunks):
                t0 = time.time()
                wav = self.model.generate(c.text, **kwargs)
                wav = _trim_leading_silence(wav, self.sr)
                gen = time.time() - t0
                audio_s = wav.shape[-1] / self.sr
                logger.info(
                    "Chatterbox chunk %d/%d [%s] gen=%.2fs audio=%.2fs rt=%.2fx",
                    i + 1, len(chunks), c.kind[:4], gen, audio_s,
                    (audio_s / gen) if gen else 0.0,
                )
                waves.append(wav)

        full = _concat_with_gaps(waves, [c.kind for c in chunks], self.sr)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(output_path), full, self.sr)
        return full.shape[-1] / self.sr, self.sr
