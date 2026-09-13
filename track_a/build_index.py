"""A1+A2 runner: corpus -> chunks -> ChromaDB index.

Usage (from repo root, inside .venv):
    python -m track_a.build_index [--skip-pdfs]
"""
from __future__ import annotations

import argparse

from shared.config import load_settings
from shared.logger import get_logger

from track_a.chunker import chunk_corpus
from track_a.corpus_ingest import ingest_corpus
from track_a.embeddings import EmbeddingBackend
from track_a.vector_store import VectorStore

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ChromaDB index from the corpus")
    parser.add_argument("--skip-pdfs", action="store_true", help="only use the seed markdown corpus")
    parser.add_argument("--backend", choices=["auto", "bge_m3", "minilm"], default=None,
                        help="override embedding backend")
    args = parser.parse_args()

    docs = ingest_corpus(use_raw_pdfs=not args.skip_pdfs)
    if not docs:
        log.warning("no documents found - nothing to index")
        return

    backend_name = args.backend or load_settings()["track_a"]["embeddings"]["backend"]
    emb = EmbeddingBackend(backend=backend_name)
    chunks = chunk_corpus(docs)
    log.info("produced %d chunks across %d documents", len(chunks), len(docs))

    store = VectorStore(embedding_backend=emb)
    store.reset()  # idempotent re-runs
    n = store.add_chunks(chunks)
    log.info("index stats: %s", store.stats())
    log.info("embedding backend used: %s", emb.backend)


if __name__ == "__main__":
    main()
