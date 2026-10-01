"""
Base TTS Adapter interface contract.
TTS: (text, lang) -> {audio_wav_bytes, ms}
"""

from abc import ABC, abstractmethod
from typing import TypedDict


class TTSResult(TypedDict):
    audio_wav_bytes: bytes
    ms: float


class BaseTTSAdapter(ABC):
    @abstractmethod
    async def synthesize(self, text: str, lang: str = "hi") -> TTSResult:
        """Synthesize text into speech WAV audio bytes."""
        pass
