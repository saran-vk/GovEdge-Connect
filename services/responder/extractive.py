"""
Extractive offline fallback responder.
Reads top RAG chunk verbatim when cloud is unreachable or in offline local mode.
"""

import time
from typing import List, Tuple
from gateway.schemas import SchemeChunk, Citation


class ExtractiveResponder:
    """Verbatim fallback responder returning top RAG chunk with degraded flag."""

    @staticmethod
    def extract_reply(context_chunks: List[SchemeChunk]) -> Tuple[str, List[Citation], float]:
        t0 = time.perf_counter()
        if not context_chunks:
            reply = "I can only answer questions regarding PM-KISAN, MGNREGA, and Ayushman Bharat."
            citations = []
        else:
            top_chunk = context_chunks[0]
            reply = top_chunk.text
            citations = [Citation(scheme=top_chunk.scheme_name, chunk_id=top_chunk.id, score=1.0)]

        ms = (time.perf_counter() - t0) * 1000
        return reply, citations, round(ms, 2)
