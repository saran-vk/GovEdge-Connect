"""Unit tests for Track B helpers (offline - no model loads)."""
import json
from pathlib import Path

from shared.contract import ASROutput
from track_b.benchmark import _load_transcripts, run_benchmark
from track_b.synth_audio import UTTERANCES


class _StubASR:
    model_size = "stub"
    device = "cpu"
    compute_type = "int8"

    def __init__(self, txn_dir: str | Path | None = None):
        self._txn_dir = Path(txn_dir) if txn_dir else Path("track_b/data/transcripts")

    def transcribe(self, wav_path: str | Path, language: str | None = None) -> ASROutput:
        clip_id = Path(wav_path).stem
        txn = self._txn_dir / f"{clip_id}.json"
        gt = json.loads(txn.read_text(encoding="utf-8"))["text"]
        return ASROutput(transcript=gt, detected_language="en", confidence_score=0.95)


def _make_clip(tmp_path: Path, clip_id: str, lang: str, text: str) -> None:
    (tmp_path / f"{clip_id}.wav").write_bytes(b"fake")
    (tmp_path / f"{clip_id}.json").write_text(
        json.dumps({"clip_id": clip_id, "language": lang, "text": text}), encoding="utf-8")


def test_synth_utterance_sets_three_languages():
    assert set(UTTERANCES) == {"en-IN", "hi-IN", "ta-IN"}
    for lang, utts in UTTERANCES.items():
        assert len(utts) >= 3


def test_transcript_loader(tmp_path):
    _make_clip(tmp_path, "en_01", "en-IN", "hello world")
    txns = _load_transcripts(tmp_path)
    assert txns["en_01"]["text"] == "hello world"


def test_wer_benchmark_perfect_asr(tmp_path):
    _make_clip(tmp_path, "en_00", "en-IN", "this is the reference text")
    result = run_benchmark(
        asr=_StubASR(tmp_path),
        audio_dir=tmp_path,
        txn_dir=tmp_path,
        results_dir=tmp_path / "results",
    )
    assert result["overall_wer"] == 0.0
    assert result["per_language"]["en-IN"]["n"] == 1
    assert (tmp_path / "results" / "wer_baseline.json").exists()


def test_benchmark_skips_missing_transcript(tmp_path):
    (tmp_path / "no_txn.wav").write_bytes(b"fake")
    result = run_benchmark(
        asr=_StubASR(), audio_dir=tmp_path, txn_dir=tmp_path,
        results_dir=tmp_path / "results",
    )
    assert result.get("rows", []) == []
