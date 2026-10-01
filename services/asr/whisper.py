"""
Real faster-whisper ASR adapter placeholder stub.
"""

from typing import Optional
from services.asr.base import BaseASRAdapter, ASRResult


class WhisperASRAdapter(BaseASRAdapter):
    """Production faster-whisper / LoRA fine-tuned ASR adapter."""

    def __init__(self, model_size_or_path: str = "base"):
        self.model_size_or_path = model_size_or_path
        self.model = None

    async def transcribe(self, wav_bytes: bytes, lang_hint: Optional[str] = None) -> ASRResult:
        raise NotImplementedError("WhisperASRAdapter requires model loading in hybrid/local mode.")
