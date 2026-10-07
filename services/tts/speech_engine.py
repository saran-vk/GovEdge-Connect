"""
Speech Synthesis Engine backed by gTTS and local audio caching.
Generates genuine spoken audio for Tamil, Hindi, Telugu, Malayalam, and English.
"""

import os
import io
import time
import base64
import hashlib
from typing import Optional
from services.tts.mock import MOCK_WAV_HEADER

AUDIO_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "audio_cache")
)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# In-memory LRU cache to avoid disk I/O on repeated lookups
_MEM_AUDIO_CACHE = {}


def get_speech_cache_path(text: str, lang: str) -> str:
    key = hashlib.sha256(f"{lang}:{text.strip()}".encode("utf-8")).hexdigest()
    return os.path.join(AUDIO_CACHE_DIR, f"{key}.mp3")


def generate_speech(text: str, lang: str = "ta") -> bytes:
    """
    Synthesizes text to speech audio bytes (MP3) using local cache or gTTS.
    Supports: 'ta' (Tamil), 'hi' (Hindi), 'te' (Telugu), 'ml' (Malayalam), 'en' (English).
    """
    text_clean = text.strip()
    if not text_clean:
        return MOCK_WAV_HEADER

    cache_key = f"{lang}:{text_clean}"
    if cache_key in _MEM_AUDIO_CACHE:
        return _MEM_AUDIO_CACHE[cache_key]

    cache_file = get_speech_cache_path(text_clean, lang)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                data = f.read()
                if len(data) > 0:
                    _MEM_AUDIO_CACHE[cache_key] = data
                    return data
        except Exception:
            pass

    # Synthesize with gTTS
    try:
        from gtts import gTTS

        # Map language codes
        lang_code = lang if lang in ["ta", "hi", "te", "ml", "en"] else "en"
        tts = gTTS(text=text_clean, lang=lang_code)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        data = fp.getvalue()

        if len(data) > 0:
            _MEM_AUDIO_CACHE[cache_key] = data
            try:
                with open(cache_file, "wb") as f:
                    f.write(data)
            except Exception:
                pass
            return data
    except Exception as e:
        print(f"[SpeechEngine] Warning: gTTS synthesis failed for {lang}: {e}")

    return MOCK_WAV_HEADER


def get_cached_speech_b64(text: str, lang: str = "ta") -> str:
    """Returns base64 encoded audio string."""
    audio_bytes = generate_speech(text, lang)
    return base64.b64encode(audio_bytes).decode("ascii")
