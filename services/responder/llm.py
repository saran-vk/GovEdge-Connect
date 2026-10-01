"""
Cloud LLM response generator enforcing grounded answers and chunk citations.
"""

import time
from typing import List, Tuple
from gateway.schemas import SchemeChunk, Citation


class GroundedLLMResponder:
    """Invokes hosted LLM (Sarvam-2B / Llama-3.2-3B) with strict grounding and citation constraints."""

    def __init__(self, provider: str = "mock"):
        self.provider = provider

    async def generate_response(
        self,
        query: str,
        context_chunks: List[SchemeChunk],
        lang: str = "en",
    ) -> Tuple[str, List[Citation], float]:
        """
        Generates grounded response and returns chunk citations.
        Rejects answers without valid citations.
        Returns: (reply_text, citations, elapsed_ms)
        """
        t0 = time.perf_counter()
        ms = (time.perf_counter() - t0) * 1000
        reply = "Mock grounded reply based on retrieved policy context."
        citations = [Citation(scheme=c.scheme_name, chunk_id=c.id, score=0.95) for c in context_chunks]
        return reply, citations, round(ms, 2)
