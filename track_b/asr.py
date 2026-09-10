"""B2 - ASR local environment: faster-whisper wrapper.

Loads a Whisper model via faster-whisper (CTranslate2) and exposes a
transcribe() that returns the standardized ASROutput contract from shared.contract.

device/compute_type are read from settings.yaml; the RTX 4050 can be enabled
later by setting device: cuda + compute_type: float16.
"""
from __future__ import annotations

import math
from pathlib import Path

from faster_whisper import WhisperModel

from shared.config import load_settings
from shared.contract import ASROutput
from shared.logger import get_logger

log = get_logger(__name__)


class ASR:
    def __init__(self, model_size: str | None = None, device: str | None = None,
                 compute_type: str | None = None):
        cfg = load_settings()["track_b"]["asr"]
        self.model_size = model_size or cfg["model_size"]
        device = device or cfg["device"]
        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        self.device = device
        self.compute_type = compute_type or cfg["compute_type"]
        log.info("loading faster-whisper %s on %s (%s)",
                 self.model_size, self.device, self.compute_type)
        self._model = WhisperModel(self.model_size, device=self.device,
                                   compute_type=self.compute_type)

    def transcribe(self, audio_path: str | Path, language: str | None = None,
                   beam_size: int | None = None) -> ASROutput:
        cfg = load_settings()["track_b"]["asr"]
        segments, info = self._model.transcribe(
            str(audio_path),
            language=language or cfg["language"],
            beam_size=beam_size or cfg["beam_size"],
            vad_filter=True,  # trims silence; reduces end-of-utterance hallucinations
        )
        seg_list = list(segments)
        text = " ".join(seg.text.strip() for seg in seg_list).strip()
        probs = [float(seg.avg_logprob) for seg in seg_list]
        confidence = float(math.exp(sum(probs) / len(probs))) if probs else 0.0
        return ASROutput(
            transcript=text or "",
            detected_language=(info.language or "unknown").lower(),
            confidence_score=min(1.0, max(0.0, confidence)),
        )


def _cuda_available() -> bool:
    """True only if a CUDA device AND the CUDA 12 runtime (cublas) are usable.

    ctranslate2's CUDA wheel bundles cudnn but expects cublas64_12.dll on the
    system PATH. If it cannot be loaded, CUDA inference would crash at encode
    time - so we treat it as unavailable and fall back to CPU.
    """
    try:
        import ctypes
        import platform

        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        if platform.system() == "Windows":
            ctypes.WinDLL("cublas64_12.dll")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    import sys

    model = ASR()
    for wav in sorted(Path("track_b/data/audio").glob("*.wav")):
        out = model.transcribe(wav)
        print(f"{wav.name:24s} [{out.detected_language}] conf={out.confidence_score:.2f} :: {out.transcript}")
    sys.exit(0)
