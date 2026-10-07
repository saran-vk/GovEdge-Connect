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
    """Mock TTS adapter providing rapid cached speech synthesis."""

    async def synthesize(self, text: str, lang: str = "hi") -> TTSResult:
        t0 = time.perf_counter()
        try:
            from services.tts.speech_engine import generate_speech
            audio_bytes = generate_speech(text, lang)
        except Exception:
            audio_bytes = MOCK_WAV_HEADER
        ms = (time.perf_counter() - t0) * 1000
        return {
            "audio_wav_bytes": audio_bytes,
            "ms": round(ms, 2),
        }
