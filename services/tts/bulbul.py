"""
Bulbul / IndicTTS adapter placeholder stub.
"""

from services.tts.base import BaseTTSAdapter, TTSResult


class BulbulTTSAdapter(BaseTTSAdapter):
    """Production TTS adapter using Bulbul/IndicTTS remote API or local model."""

    def __init__(self, api_key: str = "", voice_id: str = "default"):
        self.api_key = api_key
        self.voice_id = voice_id

    async def synthesize(self, text: str, lang: str = "hi") -> TTSResult:
        raise NotImplementedError("BulbulTTSAdapter requires API key or weights configured in hybrid mode.")
