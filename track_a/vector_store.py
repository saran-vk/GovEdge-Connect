"""A2 - Persistent ChromaDB vector store wrapper.

Indexes scheme document chunks with the configured embedding backend and
provides retrieval with the configured top_k. Backend switch (IndicBERT /
MiniLM) is transparent to callers.
"""
from __future__ import annotations

import uuid
from typing import Iterable

import chromadb
from chromadb.config import Settings

from shared.config import load_settings, resolve
from shared.logger import get_logger

from track_a.chunker import Chunk
from track_a.embeddings import EmbeddingBackend

log = get_logger(__name__)


class VectorStore:
    def __init__(self, embedding_backend: EmbeddingBackend | None = None):
        cfg = load_settings()["track_a"]
        self.cfg = cfg
        self.embeddings = embedding_backend or EmbeddingBackend()
        self.collection_name = cfg["collection_name"]
        chroma_dir = str(resolve(cfg["chroma_dir"]))
        self._client = chromadb.PersistentClient(
            path=chroma_dir, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": cfg["retrieval"]["distance"]},
        )

    def reset(self) -> None:
        """Drop and recreate the collection so re-runs are idempotent."""
        self._client.delete_collection(self.collection_name)
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.cfg["retrieval"]["distance"]},
        )
        log.info("collection '%s' reset", self.collection_name)

    def add_chunks(self, chunks: Iterable[Chunk], batch_size: int = 100) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0
        texts = [c.text for c in chunk_list]
        ids = [f"c-{uuid.uuid4().hex[:12]}" for _ in chunk_list]
        metadatas = [
            {"source": c.source, "index": c.index, **c.metadata} for c in chunk_list
        ]

        for i in range(0, len(chunk_list), batch_size):
            batch = chunk_list[i : i + batch_size]
            emb = self.embeddings.embed([c.text for c in batch])
            self.collection.add(
                ids=ids[i : i + batch_size],
                embeddings=emb.tolist(),
                documents=[c.text for c in batch],
                metadatas=metadatas[i : i + batch_size],
            )
        log.info("added %d chunks to collection '%s'", len(chunk_list), self.collection_name)
        return len(chunk_list)

    def query(self, text: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.cfg["retrieval"]["top_k"]
        vec = self.embeddings.embed([text])[0].tolist()
        res = self.collection.query(query_embeddings=[vec], n_results=k)
        hits = []
        docs = res.get("documents")
        if not docs or not docs[0]:
            log.warning("no results returned from ChromaDB for query: %s", text[:60])
            return hits
        for i, doc in enumerate(res["documents"][0]):
            hits.append(
                {
                    "text": doc,
                    "source": res["metadatas"][0][i].get("source", ""),
                    "distance": res["distances"][0][i],
                }
            )
        return hits

    def stats(self) -> dict:
        return {"count": self.collection.count(), "collection": self.collection_name}
