"""
Base ASR Adapter interface contract.
ASR: (wav_bytes, lang_hint) -> {text, lang, confidence, ms}
"""

from abc import ABC, abstractmethod
from typing import Optional, TypedDict


class ASRResult(TypedDict):
    text: str
    lang: str
    confidence: float
    ms: float


class BaseASRAdapter(ABC):
    @abstractmethod
    async def transcribe(self, wav_bytes: bytes, lang_hint: Optional[str] = None) -> ASRResult:
        """Transcribe speech audio into text."""
        pass
