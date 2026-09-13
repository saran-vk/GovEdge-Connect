"""B2 - ASR local environment: faster-whisper wrapper.

Loads a Whisper model via faster-whisper (CTranslate2) and exposes a
transcribe() that returns the standardized ASROutput contract from shared.contract.

device/compute_type are read from settings.yaml; the RTX 4050 is enabled by
default (device: auto -> cuda when usable). On hosts where only CUDA 13 is
installed (libcublas.so.13) but faster-whisper needs CUDA 12 (libcublas.so.12),
a repo-local compatibility shim (.cuda_compat/) is created automatically before
the model is loaded.
"""
from __future__ import annotations

import ctypes
import math
import os
import platform
import sys
from pathlib import Path

from faster_whisper import WhisperModel

from shared.config import load_settings
from shared.contract import ASROutput
from shared.logger import get_logger

log = get_logger(__name__)

_COMPAT_DIR = Path(__file__).resolve().parents[1] / ".cuda_compat"


def _cuda_compat_needed() -> bool:
    """True when ctranslate2 asks for libcublas.so.12 but only .13 (or none) is resolvable.

    The exact cublasLt library name is verified because ctranslate2 requires both
    libcublas and libcublasLt; a broken link otherwise only fails at inference time.
    """
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        for lib in ("libcublas.so.12", "libcublasLt.so.12"):
            try:
                ctypes.CDLL(lib)
            except OSError:
                return True
    except Exception:
        return False
    return False


def _load_lib(so_name: str) -> bool:
    try:
        ctypes.CDLL(so_name)
        return True
    except OSError:
        return False


def _ensure_cuda() -> None:
    """Prepare CUDA runtime libs and fall back to CPU when the GPU is unusable.

    On CUDA-13-only hosts the CUDA-12 connector libs faster-whisper needs
    (libcublas.so.12) do not exist; the process is re-executed once with a
    repo-local compat dir on LD_LIBRARY_PATH because glibc reads that variable
    only at process start. Set GOVCONNECT_NO_REEXEC=1 to disable re-execution.
    """
    _prepare_cuda_compat()
    if _cuda_available():
        return
    if os.environ.get("GOVCONNECT_NO_REEXEC"):
        return
    if not _COMPAT_DIR.exists() or not (sys.argv and sys.argv[0]):
        log.info("CUDA unavailable; falling back to CPU")
        return
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{_COMPAT_DIR}:" + env.get("LD_LIBRARY_PATH", "")
    env["GOVCONNECT_NO_REEXEC"] = "1"
    log.info("re-executing with CUDA compat path to enable GPU ASR")
    try:
        os.execv(sys.executable, [sys.executable, *sys.argv])
    except OSError as exc:  # pragma: no cover - exec failure is fatal
        log.error("re-exec failed (%s); falling back to CPU", exc)


def _prepare_cuda_compat() -> None:
    """Symlink a CUDA 13 cublas into the CUDA 12 names ctranslate2 expects.

    Creates `<repo>/.cuda_compat/libcublas.so.12` (and cublasLt) and prepends it
    to LD_LIBRARY_PATH / PATH before any CUDA-using operation by the process.
    No system packages are touched; fully reversible by deleting the directory.
    """
    if not _cuda_compat_needed():
        return
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            log.debug("no CUDA device available; skipping cublas compat")
            return
    except Exception:
        return

    candidates: list[Path] = []
    for extra in sys.path:
        if "site-packages" in str(extra):
            candidates += [Path(extra) / "nvidia" / "cu13" / "lib",
                           Path(extra) / "nvidia" / "cublas" / "lib"]
    for p in (Path("/opt/cuda"), Path("/usr/local/cuda"), Path("/usr/lib/ollama/cuda_v13")):
        candidates += [p / "targets" / "x86_64-linux" / "lib", p]
    found: list[Path] = []
    for base in candidates:
        found.extend(sorted(base.glob("libcublas.so.*")))
    if not found:
        log.warning("CUDA requested but no libcublas found; defaulting device selection")
        return

    src = found[0]
    _COMPAT_DIR.mkdir(parents=True, exist_ok=True)
    for stem in ("libcublas", "libcublasLt"):
        src_so = src.parent / f"{stem}.so.13"
        if not src_so.exists():
            src_so = src.parent / f"{stem}.so.12"
        dst = _COMPAT_DIR / f"{stem}.so.12"
        if src_so.exists() and not dst.exists():
            dst.unlink(missing_ok=True)
            dst.symlink_to(src_so)
        if dst.exists() and not _load_lib(str(dst)):
            log.warning("could not load %s from compat shim; ASR may fall back to CPU", dst)
    env = os.environ.get("LD_LIBRARY_PATH", "")
    if str(_COMPAT_DIR) not in env:
        os.environ["LD_LIBRARY_PATH"] = f"{_COMPAT_DIR}:{env}" if env else str(_COMPAT_DIR)
    log.info("enabled CUDA compat shim at %s (cublas %s)", _COMPAT_DIR, src.name)


def _cuda_available() -> bool:
    """True when ctranslate2 can actually run on a CUDA device."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        if platform.system() == "Windows":
            ctypes.WinDLL("cublas64_12.dll")
        else:
            ctypes.CDLL("libcublas.so.12")
        return True
    except Exception:
        return False


class ASR:
    def __init__(self, model_size: str | None = None, device: str | None = None,
                 compute_type: str | None = None):
        cfg = load_settings()["track_b"]["asr"]
        self.model_size = model_size or cfg["model_size"]
        device = device or cfg["device"]
        _ensure_cuda()
        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        self.device = device
        self.compute_type = compute_type or cfg["compute_type"]
        log.info("loading faster-whisper %s on %s (%s)",
                 self.model_size, self.device, self.compute_type)
        self._model = WhisperModel(self.model_size, device=self.device,
                                   compute_type=self.compute_type)

    def transcribe(self, audio_path: str | Path, language: str | None = None,
                   beam_size: int | None = None,
                   initial_prompt: str | None = None) -> ASROutput:
        cfg = load_settings()["track_b"]["asr"]
        audio_path = Path(audio_path)
        if not audio_path.exists():
            log.error("audio file not found: %s", audio_path)
            return ASROutput(transcript="", detected_language="unknown", confidence_score=0.0)
        segments, info = self._model.transcribe(
            str(audio_path),
            language=language or cfg["language"],
            beam_size=beam_size or cfg["beam_size"],
            vad_filter=True,  # trims silence; reduces end-of-utterance hallucinations
            initial_prompt=initial_prompt,
        )
        seg_list = list(segments)
        if not seg_list:
            log.warning("no segments produced for %s (empty audio or VAD filtered all)", audio_path.name)
            return ASROutput(transcript="", detected_language=(info.language or "unknown").lower(),
                             confidence_score=0.0)
        text = " ".join(seg.text.strip() for seg in seg_list).strip()
        probs = [float(seg.avg_logprob) for seg in seg_list]
        confidence = float(math.exp(sum(probs) / len(probs))) if probs else 0.0
        return ASROutput(
            transcript=text or "",
            detected_language=(info.language or "unknown").lower(),
            confidence_score=min(1.0, max(0.0, confidence)),
        )


def _cuda_available() -> bool:
    """True if a CUDA device is usable by ctranslate2 with working cublas.

    Checks ctranslate2 device count first, then verifies the cublas shared
    library can actually be loaded (on both Linux and Windows).
    """
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        import ctypes
        import platform

        if platform.system() == "Windows":
            ctypes.WinDLL("cublas64_12.dll")
        else:
            ctypes.CDLL("libcublas.so.12")
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
