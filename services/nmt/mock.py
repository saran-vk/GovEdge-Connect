"""
Mock NMT adapter (passthrough translation).
"""

import time
from services.nmt.base import BaseNMTAdapter, NMTResult


class MockNMTAdapter(BaseNMTAdapter):
    """Passthrough mock NMT adapter for testing without translation overhead."""

    async def translate(self, text: str, src_lang: str, tgt_lang: str) -> NMTResult:
        t0 = time.perf_counter()
        ms = (time.perf_counter() - t0) * 1000
        # In mock mode, passthrough the text
        return {
            "text": text,
            "ms": round(ms, 2),
        }
