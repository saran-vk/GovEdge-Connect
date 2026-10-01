"""
Cache-first response layer backed by Redis.
Cache key: sha256(scheme + intent + lang + corpus_version)
"""

import hashlib
import time
from typing import Optional, Tuple


class ResponseCache:
    """Pre-synthesized answer cache for common (scheme x intent x language) combinations."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url

    @staticmethod
    def generate_key(scheme: str, intent: str, lang: str, corpus_version: str = "1.0") -> str:
        payload = f"{scheme}:{intent}:{lang}:{corpus_version}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def get(self, scheme: str, intent: str, lang: str, corpus_version: str = "1.0") -> Tuple[Optional[str], Optional[str], float]:
        """
        Lookup cached response.
        Returns: (reply_text, audio_data, elapsed_ms)
        """
        t0 = time.perf_counter()
        # Stub implementation for setup
        ms = (time.perf_counter() - t0) * 1000
        return None, None, round(ms, 2)
