"""
RAG retriever service querying ChromaDB with metadata filtering.
"""

import time
from typing import List, Optional, Tuple
from gateway.schemas import Citation, SchemeChunk


class RAGRetriever:
    """Retrieves top grounded scheme chunks from ChromaDB."""

    def __init__(self, persist_dir: str = "./data/chroma"):
        self.persist_dir = persist_dir

    async def retrieve(
        self,
        query: str,
        scheme_name: Optional[str] = None,
        top_k: int = 3,
    ) -> Tuple[List[SchemeChunk], List[Citation], float]:
        """
        Queries ChromaDB vector index filtered by scheme_name.
        Returns: (chunks, citations, elapsed_ms)
        """
        t0 = time.perf_counter()
        # Mock/stub retrieval for initial setup
        chunks: List[SchemeChunk] = []
        citations: List[Citation] = []
        ms = (time.perf_counter() - t0) * 1000
        return chunks, citations, round(ms, 2)
