"""Unit tests for the podcast-mode pipeline.

Contracts tested (no GPU, no Chatterbox model, no Remotion):
  1. _validate_chunks()       — pure validation logic in src/tts_chatterbox.py
  2. _trim_leading_silence()  — pure tensor function in src/tts_chatterbox.py
  3. load_script()            — file I/O loader in scripts/render_podcast.py
  4. _write_remotion_section()— JSON/file writer in scripts/render_podcast.py
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Imports — pulled directly so we can test private helpers without loading
# the Chatterbox GPU model.
# ---------------------------------------------------------------------------
from src.tts_chatterbox import (  # noqa: E402
    HARD_MAX,
    Chunk,
    _trim_leading_silence,
    _validate_chunks,
)

# render_podcast imports _ChatterboxBackend at module level, which is fine as
# long as we never call .get() or instantiate it.  We import the module under
# a local alias and monkeypatch REMOTION_DIR in the section-writer tests.
import scripts.render_podcast as render_podcast  # noqa: E402
from scripts.render_podcast import _write_remotion_section, load_script  # noqa: E402


# ===========================================================================
# 1. _validate_chunks()
# ===========================================================================

class TestValidateChunks:
    def _chunk(self, text: str) -> list[Chunk]:
        return [Chunk(text=text, kind="paragraph")]

    def test_valid_chunk_passes(self) -> None:
        _validate_chunks(self._chunk("Everything is fine here."))

    def test_valid_chunk_with_exclamation(self) -> None:
        _validate_chunks(self._chunk("What a wonderful day!"))

    def test_valid_chunk_with_question_mark(self) -> None:
        _validate_chunks(self._chunk("Is this correct?"))

    def test_valid_chunk_ending_with_closing_quote(self) -> None:
        _validate_chunks(self._chunk('"She said hello."'))

    def test_empty_chunk_raises(self) -> None:
        with pytest.raises(ValueError, match="empty chunk"):
            _validate_chunks(self._chunk(""))

    def test_lowercase_start_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk starts mid-sentence"):
            _validate_chunks(self._chunk("lowercase start."))

    def test_no_terminal_punctuation_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk ends mid-sentence"):
            _validate_chunks(self._chunk("This ends without punctuation"))

    def test_unbalanced_quotes_raises(self) -> None:
        # Odd number of double-quote characters
        with pytest.raises(ValueError, match="unbalanced quotes"):
            _validate_chunks(self._chunk('She said "hello.'))

    def test_chunk_over_hard_max_raises(self) -> None:
        long_text = "A" + "a" * (HARD_MAX - 1) + "."   # HARD_MAX + 1 chars total
        assert len(long_text) > HARD_MAX
        with pytest.raises(ValueError, match="chunk too long"):
            _validate_chunks(self._chunk(long_text))

    def test_chunk_exactly_at_hard_max_passes(self) -> None:
        # HARD_MAX chars, ending with period, starting uppercase
        text = "A" + "a" * (HARD_MAX - 2) + "."
        assert len(text) == HARD_MAX
        _validate_chunks(self._chunk(text))

    def test_chunk_starting_with_open_bracket_passes(self) -> None:
        _validate_chunks(self._chunk("[Narrator] This is a bracket-prefixed chunk."))

    def test_chunk_starting_with_open_quote_passes(self) -> None:
        _validate_chunks(self._chunk('"Quoted opening sentence."'))

    def test_multiple_valid_chunks_pass(self) -> None:
        chunks = [
            Chunk(text="First sentence here.", kind="paragraph"),
            Chunk(text="Second sentence there.", kind="paragraph"),
        ]
        _validate_chunks(chunks)


# ===========================================================================
# 2. _trim_leading_silence()
# ===========================================================================

SR = 24_000  # common Chatterbox sample rate


class TestTrimLeadingSilence:
    def test_no_leading_silence_unchanged(self) -> None:
        wav = torch.ones(1, SR)  # loud from sample 0
        result = _trim_leading_silence(wav, SR)
        assert result.shape == wav.shape

    def test_leading_silence_gets_trimmed(self) -> None:
        # 30ms of near-zero preamble followed by loud signal
        silence_samples = int(SR * 0.030)
        silence = torch.zeros(1, silence_samples)
        signal = torch.ones(1, SR - silence_samples) * 0.1
        wav = torch.cat([silence, signal], dim=-1)

        result = _trim_leading_silence(wav, SR)

        assert result.shape[-1] < wav.shape[-1], "trim should have removed samples"

    def test_trim_does_not_exceed_max_trim_sec(self) -> None:
        max_trim_sec = 0.08
        # 200ms of silence — much more than the 80ms cap
        silence_samples = int(SR * 0.200)
        silence = torch.zeros(1, silence_samples)
        signal = torch.ones(1, SR) * 0.1
        wav = torch.cat([silence, signal], dim=-1)

        result = _trim_leading_silence(wav, SR, max_trim_sec=max_trim_sec)

        trimmed = wav.shape[-1] - result.shape[-1]
        max_allowed = int(SR * max_trim_sec)
        assert trimmed <= max_allowed, (
            f"trimmed {trimmed} samples but cap is {max_allowed}"
        )

    def test_works_on_1d_tensor(self) -> None:
        silence_samples = int(SR * 0.030)
        wav_1d = torch.cat([
            torch.zeros(silence_samples),
            torch.ones(SR - silence_samples) * 0.1,
        ])
        assert wav_1d.dim() == 1

        result = _trim_leading_silence(wav_1d, SR)

        # Should not crash and should trim or keep depending on threshold
        assert result.dim() == 1

    def test_works_on_2d_tensor(self) -> None:
        silence_samples = int(SR * 0.030)
        wav_2d = torch.cat([
            torch.zeros(1, silence_samples),
            torch.ones(1, SR - silence_samples) * 0.1,
        ], dim=-1)
        assert wav_2d.dim() == 2

        result = _trim_leading_silence(wav_2d, SR)

        assert result.dim() == 2

    def test_all_silence_returns_something(self) -> None:
        wav = torch.zeros(1, SR)
        result = _trim_leading_silence(wav, SR)
        # Should not crash; returns at most the original tensor
        assert result.shape[-1] <= wav.shape[-1]


# ===========================================================================
# 3. load_script()
# ===========================================================================

class TestLoadScript:
    def _write_script(self, tmp_path: Path, body: str) -> Path:
        p = tmp_path / "test_script.py"
        p.write_text(body)
        return p

    def test_basic_title_and_script(self, tmp_path: Path) -> None:
        script_file = self._write_script(tmp_path, """
TITLE = "my_cool_topic"
SCRIPT = [("heart", "Hello world.", 0.7)]
""")
        title, script, meta = load_script(script_file)
        assert title == "my_cool_topic"
        assert script == [("heart", "Hello world.", 0.7)]

    def test_default_speaker_map(self, tmp_path: Path) -> None:
        script_file = self._write_script(tmp_path, """
TITLE = "topic"
SCRIPT = []
""")
        _, _, meta = load_script(script_file)
        assert meta["speaker_map"] == {"heart": "maya", "fenrir": "marcus"}

    def test_default_section_id(self, tmp_path: Path) -> None:
        script_file = self._write_script(tmp_path, """
TITLE = "topic"
SCRIPT = []
""")
        _, _, meta = load_script(script_file)
        assert meta["section_id"] == "section_00"

    def test_default_title_label_derived_from_title(self, tmp_path: Path) -> None:
        script_file = self._write_script(tmp_path, """
TITLE = "my_cool_topic"
SCRIPT = []
""")
        _, _, meta = load_script(script_file)
        # title.replace("_", " ").title() => "My Cool Topic"
        assert meta["title_label"] == "My Cool Topic"

    def test_custom_speaker_map_override(self, tmp_path: Path) -> None:
        script_file = self._write_script(tmp_path, """
TITLE = "topic"
SCRIPT = []
SPEAKER_MAP = {"heart": "alice", "fenrir": "bob"}
""")
        _, _, meta = load_script(script_file)
        assert meta["speaker_map"] == {"heart": "alice", "fenrir": "bob"}

    def test_custom_section_id_override(self, tmp_path: Path) -> None:
        script_file = self._write_script(tmp_path, """
TITLE = "topic"
SCRIPT = []
SECTION_ID = "section_01"
""")
        _, _, meta = load_script(script_file)
        assert meta["section_id"] == "section_01"

    def test_custom_title_label_override(self, tmp_path: Path) -> None:
        script_file = self._write_script(tmp_path, """
TITLE = "topic"
SCRIPT = []
TITLE_LABEL = "My Custom Label"
""")
        _, _, meta = load_script(script_file)
        assert meta["title_label"] == "My Custom Label"

    def test_title_defaults_to_stem_when_missing(self, tmp_path: Path) -> None:
        script_file = self._write_script(tmp_path, """
SCRIPT = []
""")
        title, _, meta = load_script(script_file)
        assert title == "test_script"
        # title_label should be derived from the stem
        assert meta["title_label"] == "Test Script"


# ===========================================================================
# 4. _write_remotion_section()
# ===========================================================================

class TestWriteRemotion:
    """Monkeypatch REMOTION_DIR so nothing is written to the real remotion/ tree."""

    MOCK_TIMINGS = [
        {
            "turn": 0,
            "speaker": "heart",
            "line": "Hello from heart.",
            "start": 0.0,
            "duration": 2.345678,
            "exag": 0.7,
            "gen_sec": 1.1,
        },
        {
            "turn": 1,
            "speaker": "fenrir",
            "line": "Hello from fenrir.",
            "start": 2.545678,
            "duration": 1.123456,
            "exag": 0.7,
            "gen_sec": 0.9,
        },
    ]

    META = {
        "speaker_map": {"heart": "maya", "fenrir": "marcus"},
        "section_id": "section_00",
        "title_label": "Test Podcast",
    }

    def _dummy_wav(self, tmp_path: Path) -> Path:
        wav_path = tmp_path / "dummy.wav"
        # Minimal RIFF header stub — enough for shutil.copy2
        wav_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        return wav_path

    def test_json_written_to_correct_path(self, tmp_path: Path, monkeypatch) -> None:
        remotion_dir = tmp_path / "remotion"
        monkeypatch.setattr(render_podcast, "REMOTION_DIR", remotion_dir)

        wav = self._dummy_wav(tmp_path)
        _write_remotion_section("test", self.META, self.MOCK_TIMINGS, wav)

        expected = remotion_dir / "src" / "data" / "section_00.json"
        assert expected.exists(), f"JSON not found at {expected}"

    def test_json_has_required_keys(self, tmp_path: Path, monkeypatch) -> None:
        remotion_dir = tmp_path / "remotion"
        monkeypatch.setattr(render_podcast, "REMOTION_DIR", remotion_dir)

        wav = self._dummy_wav(tmp_path)
        _write_remotion_section("test", self.META, self.MOCK_TIMINGS, wav)

        json_path = remotion_dir / "src" / "data" / "section_00.json"
        data = json.loads(json_path.read_text())

        assert "title" in data
        assert "audioFile" in data
        assert "turns" in data

    def test_json_title_is_title_label(self, tmp_path: Path, monkeypatch) -> None:
        remotion_dir = tmp_path / "remotion"
        monkeypatch.setattr(render_podcast, "REMOTION_DIR", remotion_dir)

        wav = self._dummy_wav(tmp_path)
        _write_remotion_section("test", self.META, self.MOCK_TIMINGS, wav)

        data = json.loads(
            (remotion_dir / "src" / "data" / "section_00.json").read_text()
        )
        assert data["title"] == "Test Podcast"

    def test_json_audio_file_references_section_id(self, tmp_path: Path, monkeypatch) -> None:
        remotion_dir = tmp_path / "remotion"
        monkeypatch.setattr(render_podcast, "REMOTION_DIR", remotion_dir)

        wav = self._dummy_wav(tmp_path)
        _write_remotion_section("test", self.META, self.MOCK_TIMINGS, wav)

        data = json.loads(
            (remotion_dir / "src" / "data" / "section_00.json").read_text()
        )
        assert data["audioFile"] == "section_00.wav"

    def test_turns_have_required_keys(self, tmp_path: Path, monkeypatch) -> None:
        remotion_dir = tmp_path / "remotion"
        monkeypatch.setattr(render_podcast, "REMOTION_DIR", remotion_dir)

        wav = self._dummy_wav(tmp_path)
        _write_remotion_section("test", self.META, self.MOCK_TIMINGS, wav)

        data = json.loads(
            (remotion_dir / "src" / "data" / "section_00.json").read_text()
        )
        for turn in data["turns"]:
            assert "speaker" in turn
            assert "line" in turn
            assert "start" in turn
            assert "duration" in turn

    def test_speaker_mapped_correctly(self, tmp_path: Path, monkeypatch) -> None:
        remotion_dir = tmp_path / "remotion"
        monkeypatch.setattr(render_podcast, "REMOTION_DIR", remotion_dir)

        wav = self._dummy_wav(tmp_path)
        _write_remotion_section("test", self.META, self.MOCK_TIMINGS, wav)

        data = json.loads(
            (remotion_dir / "src" / "data" / "section_00.json").read_text()
        )
        assert data["turns"][0]["speaker"] == "maya"
        assert data["turns"][1]["speaker"] == "marcus"

    def test_start_and_duration_rounded_to_3dp(self, tmp_path: Path, monkeypatch) -> None:
        remotion_dir = tmp_path / "remotion"
        monkeypatch.setattr(render_podcast, "REMOTION_DIR", remotion_dir)

        wav = self._dummy_wav(tmp_path)
        _write_remotion_section("test", self.META, self.MOCK_TIMINGS, wav)

        data = json.loads(
            (remotion_dir / "src" / "data" / "section_00.json").read_text()
        )
        for turn in data["turns"]:
            # round(x, 3) must equal the stored value
            assert turn["start"] == round(turn["start"], 3)
            assert turn["duration"] == round(turn["duration"], 3)

    def test_wav_copied_to_public_dir(self, tmp_path: Path, monkeypatch) -> None:
        remotion_dir = tmp_path / "remotion"
        monkeypatch.setattr(render_podcast, "REMOTION_DIR", remotion_dir)

        wav = self._dummy_wav(tmp_path)
        _write_remotion_section("test", self.META, self.MOCK_TIMINGS, wav)

        dst = remotion_dir / "public" / "section_00.wav"
        assert dst.exists(), f"Wav not copied to {dst}"
        assert dst.read_bytes() == wav.read_bytes()

    def test_returns_list_of_turn_dicts(self, tmp_path: Path, monkeypatch) -> None:
        remotion_dir = tmp_path / "remotion"
        monkeypatch.setattr(render_podcast, "REMOTION_DIR", remotion_dir)

        wav = self._dummy_wav(tmp_path)
        turns = _write_remotion_section("test", self.META, self.MOCK_TIMINGS, wav)

        assert isinstance(turns, list)
        assert len(turns) == len(self.MOCK_TIMINGS)
        for turn in turns:
            assert isinstance(turn, dict)

    def test_custom_section_id_used_for_paths(self, tmp_path: Path, monkeypatch) -> None:
        remotion_dir = tmp_path / "remotion"
        monkeypatch.setattr(render_podcast, "REMOTION_DIR", remotion_dir)

        meta = {**self.META, "section_id": "section_42"}
        wav = self._dummy_wav(tmp_path)
        _write_remotion_section("test", meta, self.MOCK_TIMINGS, wav)

        assert (remotion_dir / "src" / "data" / "section_42.json").exists()
        assert (remotion_dir / "public" / "section_42.wav").exists()
