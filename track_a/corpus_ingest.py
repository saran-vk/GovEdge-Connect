"""A1 - Document corpus collection & structuring.

Ingests e-Gov welfare documents into cleaned plain text:
  * markdown fact sheets under track_a/data/corpus (seed corpus)
  * PDFs dropped into track_a/data/raw (pdfplumber, with pypdf fallback)

Output: {source_path: cleaned_text} dict, optionally persisted to
track_a/data/processed/ as .txt mirrors.
"""
from __future__ import annotations

import re
from pathlib import Path

from shared.config import load_settings, resolve
from shared.logger import get_logger

log = get_logger(__name__)


def _clean(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def read_markdown(path: Path) -> str:
    return _clean(path.read_text(encoding="utf-8"))


def read_pdf(path: Path) -> str:
    """Extract text from a PDF using pdfplumber, falling back to pypdf."""
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return _clean("\n\n".join(page.extract_text() or "" for page in pdf.pages))
    except Exception as exc:  # pdfplumber failure -> pypdf
        log.warning("pdfplumber failed for %s (%s); trying pypdf", path.name, exc)
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return _clean("\n\n".join(pages))


def ingest_corpus(use_raw_pdfs: bool = True) -> dict[str, str]:
    """Collect cleaned text from the seed corpus (and any raw PDFs)."""
    cfg = load_settings()["track_a"]
    corpus_dir = resolve(cfg["corpus_dir"])
    raw_dir = resolve(cfg["raw_dir"])
    processed_dir = resolve(cfg["processed_dir"])

    docs: dict[str, str] = {}
    for md_path in sorted(corpus_dir.glob("*.md")):
        docs[str(md_path)] = read_markdown(md_path)

    if use_raw_pdfs:
        for pdf_path in sorted(raw_dir.glob("*.pdf")):
            docs[str(pdf_path)] = read_pdf(pdf_path)

    processed_dir.mkdir(parents=True, exist_ok=True)
    for source, text in docs.items():
        out = processed_dir / f"{Path(source).stem}.txt"
        out.write_text(text, encoding="utf-8")

    log.info("ingested %d documents (%d from corpus, %d pdfs)",
             len(docs), sum(1 for s in docs if s.endswith(".md")), sum(1 for s in docs if s.endswith(".pdf")))
    return docs


if __name__ == "__main__":
    ingest_corpus()
