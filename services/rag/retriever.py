"""
RAG retriever service querying ChromaDB with metadata filtering,
dense similarity search, and lexical re-ranking for maximum recall.
"""

import time
from typing import List, Optional, Tuple
import chromadb

from gateway.schemas import Citation, SchemeChunk
from services.rag.ingest import HybridFastEmbeddingFunction, tokenize_stem


class RAGRetriever:
    """Retrieves top grounded scheme chunks from ChromaDB using hybrid search."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = "govconnect_schemes",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_fn = HybridFastEmbeddingFunction()

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    async def retrieve(
        self,
        query: str,
        scheme_name: Optional[str] = None,
        top_k: int = 3,
    ) -> Tuple[List[SchemeChunk], List[Citation], float]:
        """
        Queries ChromaDB with candidate retrieval and lexical re-ranking.
        Returns: (chunks, citations, elapsed_ms)
        """
        t0 = time.perf_counter()

        where_filter = {"scheme_name": scheme_name} if scheme_name else None

        try:
            # Query a candidate pool (up to 10 chunks per scheme)
            results = self.collection.query(
                query_texts=[query],
                n_results=10,
                where=where_filter,
            )
        except Exception:
            ms = (time.perf_counter() - t0) * 1000
            return [], [], round(ms, 2)

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else [0.2] * len(ids)

        if not ids:
            ms = (time.perf_counter() - t0) * 1000
            return [], [], round(ms, 2)

        # Hybrid lexical + dense re-ranking
        q_tokens = set(tokenize_stem(query))
        scored_candidates = []

        for i, chunk_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc_text = documents[i] if i < len(documents) else ""
            dist = distances[i] if i < len(distances) else 0.2
            dense_sim = max(0.0, 1.0 - dist)

            d_tokens = set(tokenize_stem(doc_text))
            lexical_overlap = len(q_tokens.intersection(d_tokens))

            # Final hybrid score
            hybrid_score = dense_sim + 0.35 * lexical_overlap

            sch_name = meta.get("scheme_name", scheme_name or "unknown")
            chunk = SchemeChunk(
                id=chunk_id,
                text=doc_text,
                scheme_name=sch_name,
                eligibility=None,
                state=None,
                source_doc=meta.get("source_doc"),
                page=None,
                lang=meta.get("lang", "en"),
                corpus_version=meta.get("corpus_version", "1.0"),
            )
            citation = Citation(
                scheme=sch_name,
                chunk_id=chunk_id,
                score=round(min(1.0, dense_sim), 3),
            )
            scored_candidates.append((hybrid_score, chunk, citation))

        # Sort by hybrid score descending and pick top_k
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:top_k]

        chunks = [c[1] for c in top_candidates]
        citations = [c[2] for c in top_candidates]

        ms = (time.perf_counter() - t0) * 1000
        return chunks, citations, round(ms, 2)
