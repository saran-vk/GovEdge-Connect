"""
RAG document ingestion pipeline for welfare scheme guidelines into ChromaDB.
"""

from typing import List
from gateway.schemas import SchemeChunk


class SchemeIngester:
    """Chunks policy documents and embeds them into ChromaDB using BAAI/bge-m3."""

    def __init__(self, persist_dir: str = "./data/chroma"):
        self.persist_dir = persist_dir

    def ingest_document(self, file_path: str, scheme_name: str, lang: str = "en") -> List[SchemeChunk]:
        """Reads document, applies RecursiveCharacterTextSplitter (~500/50), persists to ChromaDB."""
        # Stub to be implemented in Phase 2
        return []
