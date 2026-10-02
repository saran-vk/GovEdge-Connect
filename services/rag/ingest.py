"""
RAG document ingestion pipeline for welfare scheme guidelines into ChromaDB.
Chunks policy documents and indexes them into ChromaDB vector store with Stemmed IDF-weighted embeddings.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from gateway.schemas import SchemeChunk

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of",
    "and", "or", "with", "by", "from", "what", "who", "how", "when", "where", "which",
    "why", "can", "could", "do", "does", "did", "i", "my", "me", "we", "our", "you",
    "your", "be", "been", "being", "have", "has", "had", "if", "this", "that", "these",
    "those", "it", "its", "as",
}

IDF_PATH = Path("./data/chroma/idf_weights.json")


def tokenize_stem(text: str) -> List[str]:
    """Normalizes, tokenizes, and stems words for accurate robust lexical embedding."""
    text = text.lower()
    text = re.sub(r"rs\.?\s*", "rupees ", text)
    text = re.sub(r"₹\s*", "rupees ", text)
    text = re.sub(r"(\d+),(\d+)", r"\1\2", text)
    text = re.sub(r"(\d+),(\d+)", r"\1\2", text)
    words = re.findall(r"[a-zA-Z0-9]+", text)
    stems = []
    for w in words:
        if w in STOPWORDS:
            continue
        for suf in ["ing", "ed", "ers", "er", "es", "s"]:
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        stems.append(w)
    return stems


class HybridFastEmbeddingFunction(EmbeddingFunction):
    """
    Fast, deterministic dense embedding function (768-dim) with Stemmed IDF term weighting.
    Uses subword n-gram hashing and term weighting with L2 normalization.
    Runs instantaneously on CPU / edge without requiring external model downloads.
    """

    def __init__(self, dim: int = 768, idf_weights: Optional[Dict[str, float]] = None):
        self.dim = dim
        self.idf_weights = idf_weights or {}
        if not self.idf_weights and IDF_PATH.exists():
            try:
                self.idf_weights = json.loads(IDF_PATH.read_text(encoding="utf-8"))
            except Exception:
                self.idf_weights = {}

    def name(self) -> str:
        return "hybrid_fast_stem_idf"

    def set_idf(self, idf: Dict[str, float]):
        self.idf_weights = idf

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: List[List[float]] = []
        for text in input:
            vec = np.zeros(self.dim, dtype=np.float32)
            tokens = tokenize_stem(text)
            for t in tokens:
                w_weight = self.idf_weights.get(t, 1.0)
                idx = (hash(t) & 0x7FFFFFFF) % self.dim
                vec[idx] += w_weight
                for n in (3, 4):
                    if len(t) >= n:
                        for i in range(len(t) - n + 1):
                            ng = t[i : i + n]
                            n_idx = (hash(ng) & 0x7FFFFFFF) % self.dim
                            vec[n_idx] += 0.3 * w_weight
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings


class SchemeIngester:
    """Chunks policy documents and embeds them into ChromaDB."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = "govconnect_schemes",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_fn = HybridFastEmbeddingFunction()

        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)

    def _split_markdown_sections(self, content: str, scheme_name: str, max_chunk_len: int = 600) -> List[Dict[str, str]]:
        """Splits markdown cleanly by sections and coherent paragraphs."""
        raw_sections = re.split(r"\n(?=#{2,3}\s+)", content)
        chunks: List[Dict[str, str]] = []

        for sec in raw_sections:
            sec = sec.strip()
            if not sec:
                continue

            lines = sec.split("\n")
            header = lines[0].lstrip("#").strip()
            body = "\n".join(lines[1:]).replace("---", "").strip()

            if len(body) > max_chunk_len:
                paras = [p.strip() for p in body.split("\n\n") if p.strip()]
                for idx, p in enumerate(paras):
                    chunks.append({
                        "scheme": scheme_name,
                        "header": f"{header} (Part {idx + 1})",
                        "text": p,
                    })
            else:
                chunks.append({
                    "scheme": scheme_name,
                    "header": header,
                    "text": body or header,
                })

        return chunks

    def ingest_all(self, schemes_dir: str = "./data/schemes") -> Dict[str, int]:
        """Ingests all scheme documents, computes corpus IDF, and stores into ChromaDB."""
        files = {
            "pm_kisan": "pm_kisan.md",
            "mgnrega": "mgnrega.md",
            "ayushman_bharat": "ayushman_bharat.md",
        }

        all_doc_entries = []
        scheme_chunks_map: Dict[str, List[SchemeChunk]] = {}

        for scheme_name, filename in files.items():
            fpath = os.path.join(schemes_dir, filename)
            if not os.path.exists(fpath):
                continue

            content = Path(fpath).read_text(encoding="utf-8")
            raw_chunks = self._split_markdown_sections(content, scheme_name=scheme_name)

            chunks: List[SchemeChunk] = []
            for idx, item in enumerate(raw_chunks, 1):
                chunk_id = f"{scheme_name}_chunk_{idx:02d}"
                doc_str = f"[{scheme_name}] {item['header']}: {item['text']}"
                chunks.append(
                    SchemeChunk(
                        id=chunk_id,
                        text=doc_str,
                        scheme_name=scheme_name,
                        eligibility=None,
                        state=None,
                        source_doc=filename,
                        page=None,
                        lang="en",
                        corpus_version="1.0",
                    )
                )
                all_doc_entries.append((chunk_id, doc_str, scheme_name, item["header"], filename))
            scheme_chunks_map[scheme_name] = chunks

        # Compute Corpus DF using stem tokens
        total_docs = len(all_doc_entries)
        df: Dict[str, int] = {}
        for _, text, _, _, _ in all_doc_entries:
            tokens = set(tokenize_stem(text))
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        idf = {w: round(math.log(1.0 + total_docs / count), 4) for w, count in df.items()}
        IDF_PATH.parent.mkdir(parents=True, exist_ok=True)
        IDF_PATH.write_text(json.dumps(idf, indent=2), encoding="utf-8")
        self.embedding_fn.set_idf(idf)

        # Re-create collection with updated embedding function
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [e[0] for e in all_doc_entries]
        docs = [e[1] for e in all_doc_entries]
        metas = [
            {
                "scheme_name": e[2],
                "header": e[3],
                "source_doc": e[4],
                "lang": "en",
                "corpus_version": "1.0",
            }
            for e in all_doc_entries
        ]

        collection.add(ids=ids, documents=docs, metadatas=metas)
        results = {k: len(v) for k, v in scheme_chunks_map.items()}
        print("Ingestion complete with Stemmed IDF weighting:", results)
        return results


def main():
    ingester = SchemeIngester()
    results = ingester.ingest_all()
    print("Ingestion complete:", results)


if __name__ == "__main__":
    main()
