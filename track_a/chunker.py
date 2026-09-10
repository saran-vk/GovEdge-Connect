"""A1 - Recursive text chunking tuned for retrieval precision.

Implements the plan's parameters: chunk_size 512 / overlap 64 (configurable).
Uses langchain_core's RecursiveCharacterTextSplitter with paragraph/sentence
separators, then records per-chunk metadata (source file, chunk index).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.config import load_settings


@dataclass
class Chunk:
    text: str
    source: str
    index: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "index": self.index,
            "metadata": self.metadata,
        }


def _separators() -> list[str]:
    return ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    cfg = load_settings()["track_a"]["chunking"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or cfg["chunk_size"],
        chunk_overlap=overlap or cfg["overlap"],
        separators=_separators(),
    )
    return [c.strip() for c in splitter.split_text(text) if c and c.strip()]


def chunk_document(text: str, source: str) -> list[Chunk]:
    pieces = chunk_text(text)
    return [Chunk(text=p, source=source, index=i) for i, p in enumerate(pieces)]


def chunk_corpus(docs: dict[str, str]) -> list[Chunk]:
    """docs: {source_path: cleaned_text}. Returns all chunks with metadata."""
    chunks: list[Chunk] = []
    for source, text in docs.items():
        for c in chunk_document(text, source):
            c.metadata["source_name"] = Path(source).name
            chunks.append(c)
    return chunks
