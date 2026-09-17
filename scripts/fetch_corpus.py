"""Week-2 corpus fetcher: downloads official e-Gov welfare scheme documents.

Downloads the curated manifest of official Government-of-India PDFs (PIB /
pmjay.gov.in / s3waas) into track_a/data/raw/, which corpus_ingest.py picks up
automatically via pdfplumber.

Usage (from repo root, inside .venv):
    python -m scripts.fetch_corpus --list          # show the manifest
    python -m scripts.fetch_corpus --fetch         # download all missing
    python -m scripts.fetch_corpus --url <URL>     # add one ad-hoc PDF
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import requests  # noqa: PLC0415 (runtime dep of this script only)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("fetch_corpus")

# name -> (official source, target filename). Keep URLs from official
# Government of India domains (pib.gov.in, *.nic.in, s3waas.gov.in …).
MANIFEST: dict[str, tuple[str, str]] = {
    # --- PM-JAY (Ayushman Bharat) ---
    "pmjay_beneficiary_empowerment": (
        "https://cdnbbsr.s3waas.gov.in/s3169779d3852b32ce8b1a1724dbf5217d/uploads/2022/05/2022051067.pdf",
        "pmjay_beneficiary_empowerment.pdf",
    ),
    "pmjay_beneficiary_id_guidelines": (
        "https://cdnbbsr.s3waas.gov.in/s3169779d3852b32ce8b1a1724dbf5217d/uploads/2022/05/2022051092.pdf",
        "pmjay_beneficiary_identification.pdf",
    ),
    "pmjay_hbp_user_guidelines": (
        "https://cdnbbsr.s3waas.gov.in/s3169779d3852b32ce8b1a1724dbf5217d/uploads/2024/09/202409251770905432.pdf",
        "pmjay_hbp_user_guidelines_2.0.pdf",
    ),
    "pmjay_empanelment_guidelines": (
        "https://cdnbbsr.s3waas.gov.in/s3169779d3852b32ce8b1a1724dbf5217d/uploads/2024/08/20240828153687657.pdf",
        "pmjay_hospital_empanelment.pdf",
    ),
    # --- PMAY (Rural housing) ---
    "pmayg_factsheet_2024": (
        "https://static.pib.gov.in/WriteReadData/specificdocs/documents/2024/nov/doc20241119437801.pdf",
        "pmayg_factsheet_2024.pdf",
    ),
    "pmayg_door_to_dignity": (
        "https://static.pib.gov.in/WriteReadData/specificdocs/documents/2021/sep/PMAY%20Eng.pdf",
        "pmayg_door_to_dignity.pdf",
    ),
}


def fetch_one(url: str, dest: Path, timeout: int = 60) -> bool:
    log.info("fetching %s", url)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    if not r.content[:4] == b"%PDF":
        log.warning("skipped %s: response is not a PDF", url)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    log.info("saved %s (%.1f kB)", dest.name, len(r.content) / 1024)
    return True


def fetch_manifest(raw_dir: Path, only: str | None = None) -> list[Path]:
    saved: list[Path] = []
    for name, (url, fname) in MANIFEST.items():
        if only and name != only:
            continue
        dest = raw_dir / fname
        if dest.exists():
            log.info("exists - skip %s", fname)
            saved.append(dest)
            continue
        try:
            if fetch_one(url, dest):
                saved.append(dest)
        except requests.RequestException as exc:
            log.warning("failed %s: %s", name, exc)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print manifest entries")
    parser.add_argument("--fetch", action="store_true", help="download all missing PDFs")
    parser.add_argument("--only", default=None, help="fetch a single manifest key")
    parser.add_argument("--url", default=None, help="fetch one ad-hoc URL into raw/")
    args = parser.parse_args()

    if args.list:
        for name, (url, fname) in MANIFEST.items():
            print(f"{name:32s} {fname:40s} {url}")
        return

    raw_dir = Path("track_a/data/raw")
    if args.url:
        dest = raw_dir / Path(args.url.split("/")[-1]).name
        fetch_one(args.url, dest)
        return

    if args.fetch:
        saved = fetch_manifest(raw_dir, args.only)
        log.info("done: %d pdf(s) in %s", len(saved), raw_dir)
        return

    parser.print_help()


if __name__ == "__main__":
    sys.exit(main())