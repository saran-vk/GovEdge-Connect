"""
Mock TTS adapter returning minimal valid WAV header bytes.
"""

import time
from services.tts.base import BaseTTSAdapter, TTSResult

# Minimal 44-byte WAV header for silent 16kHz mono audio
MOCK_WAV_HEADER = (
    b"RIFF\x2c\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x08\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00"
)


class MockTTSAdapter(BaseTTSAdapter):
    """Mock TTS adapter for quick dev and testing."""

    async def synthesize(self, text: str, lang: str = "hi") -> TTSResult:
        t0 = time.perf_counter()
        ms = (time.perf_counter() - t0) * 1000
        return {
            "audio_wav_bytes": MOCK_WAV_HEADER,
            "ms": round(ms, 2),
        }
