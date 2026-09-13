"""End-to-end demo: audio clip -> ASR -> contract -> intent+entities -> RAG top-k -> response.

Usage (from repo root, inside .venv):
    python -m demo.end_to_end [clip_path] [--text "optional raw text"]
"""
from __future__ import annotations

import argparse

from shared.contract import NLUQuery, validate_against_schema
from shared.logger import get_logger

from track_a.intent_classifier import IntentClassifier
from track_a.responder import generate_response
from track_a.vector_store import VectorStore
from track_b.asr import ASR

log = get_logger(__name__)


def run(clip_path: str | None = None, text: str | None = None) -> None:
    print("=" * 72)
    print("WELFARE ASSISTANT - end-to-end demo (Week 1 prototype)")
    print("=" * 72)

    asr = ASR()
    nlu = IntentClassifier()
    store = VectorStore()

    if text is not None:
        asr_out = None
        query_text = text
        print(f"\n[0] Using provided text (no ASR): {text}")
    else:
        asr_out = asr.transcribe(clip_path)
        query_text = asr_out.transcript
        print(f"\n[0] ASR  -> transcript={query_text!r}")
        print(f"          language={asr_out.detected_language} "
              f"confidence={asr_out.confidence_score:.3f}")
        validate_against_schema(asr_out.model_dump(), "asr_output")
        print("          [contract: ASROutput validated OK]")

    analysis = nlu.analyse(query_text)
    print(f"[1] NLU  -> intent={analysis['intent']}")
    print(f"          entities={analysis['entities']}")

    hits = store.query(query_text)
    print(f"[2] RAG  -> top {len(hits)} chunks from ChromaDB:")
    for i, hit in enumerate(hits, 1):
        print(f"          {i}. [{hit['source']}] d={hit['distance']:.4f} {hit['text'][:80]}...")

    response = generate_response(
        intent=analysis["intent"],
        entities=analysis["entities"],
        chunks=hits,
        query=query_text,
    )
    print(f"\n[3] Response:\n")
    for line in response.split("\n"):
        print(f"    {line}")

    nlu_query = NLUQuery(
        query=query_text,
        language=(asr_out.detected_language if asr_out else "text"),
        intents=[analysis["intent"]],
        entities=analysis["entities"],
        sources=[h["source"] for h in hits],
    )
    validate_against_schema(nlu_query.model_dump(), "nlu_query")
    print(f"\n[4] Contract NLUQuery: intents={nlu_query.intents} sources={len(nlu_query.sources)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 1 end-to-end demo")
    parser.add_argument("clip", nargs="?", default=None, help="path to a wav clip")
    parser.add_argument("--text", default=None, help="skip ASR and use this text directly")
    args = parser.parse_args()

    if args.clip is None and args.text is None:
        parser.error("provide a wav clip path or --text")
    run(args.clip, args.text)


if __name__ == "__main__":
    main()
