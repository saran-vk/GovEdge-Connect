"""
Mock ASR adapter returning canned transcripts by audio hash or mock text.
"""

import time
import hashlib
from typing import Optional
from services.asr.base import BaseASRAdapter, ASRResult


class MockASRAdapter(BaseASRAdapter):
    """Mock ASR adapter for testing without GPU/model overhead."""

    def __init__(self, default_text: str = "PM KISAN yojana apply eligibility"):
        self.default_text = default_text

    async def transcribe(self, wav_bytes: bytes, lang_hint: Optional[str] = None) -> ASRResult:
        t0 = time.perf_counter()
        lang = lang_hint or "hi"
        # Deterministic dummy transcription
        text = self.default_text
        ms = (time.perf_counter() - t0) * 1000
        return {
            "text": text,
            "lang": lang,
            "confidence": 0.95,
            "ms": round(ms, 2),
        }
