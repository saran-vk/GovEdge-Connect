"""Unit tests for chunking, taxonomy, corpus ingestion, and response generation (offline)."""
from pathlib import Path

import pytest

from shared.contract import ASROutput, NLUQuery, example_asr_output, validate_against_schema
from track_a.chunker import chunk_corpus, chunk_document, chunk_text
from track_a.corpus_ingest import read_markdown
from track_a.responder import generate_response
from track_a.taxonomy import ENTITIES, INTENTS

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"


@pytest.fixture(scope="module")
def sample_text() -> str:
    return (
        "Ayushman Bharat provides Rs. 5 lakh cover per family.\n\n"
        "Eligibility is decided by the SECC 2011 database.\n\n"
        "Applicants need a ration card and an Aadhaar card."
    )


class TestChunking:
    def test_respects_chunk_size(self, sample_text):
        chunks = chunk_text(sample_text, chunk_size=64, overlap=8)
        assert chunks
        assert all(len(c) <= 72 for c in chunks)  # 64 + overlap margin

    def test_chunk_metadata(self, sample_text):
        chunks = chunk_document(sample_text, "pmjay.md")
        assert chunks[0].source == "pmjay.md"
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_chunk_corpus_sets_source_name(self, sample_text):
        chunks = chunk_corpus({"scheme_a.md": sample_text})
        assert chunks[0].metadata["source_name"] == "scheme_a.md"

    def test_splits_on_sections(self, sample_text):
        chunks = chunk_text(sample_text, chunk_size=60, overlap=4)
        assert len(chunks) >= 2  # paragraph splitter kicks in


class TestTaxonomy:
    def test_five_intents(self):
        assert set(INTENTS) == {
            "check_eligibility",
            "scheme_benefits",
            "required_documents",
            "application_status",
            "general_inquiry",
        }

    def test_six_entities(self):
        assert set(ENTITIES) == {
            "scheme_name",
            "age",
            "income_level",
            "occupation",
            "caste_category",
            "district_state",
        }


class TestCorpusIngest:
    def test_corpus_has_seed_sheets(self):
        sheets = list(CORPUS_DIR.glob("*.md"))
        assert len(sheets) >= 8  # 4 original + 4 expanded

    def test_markdown_readable(self):
        for path in CORPUS_DIR.glob("*.md"):
            text = read_markdown(path)
            assert len(text) > 200
            assert "\r" not in text


class TestResponder:
    def test_generates_response_with_chunks(self):
        chunks = [{"text": "PM-KISAN gives Rs. 6000 per year to farmers.", "source": "pm_kisan.md", "distance": 0.5}]
        response = generate_response("check_eligibility", {"scheme_name": ["pm-kisan"]}, chunks)
        assert "PM-KISAN" in response or "pm-kisan" in response.lower()
        assert "Rs. 6000" in response

    def test_generates_response_without_chunks(self):
        response = generate_response("general_inquiry", {}, [])
        assert "No specific information" in response or "the requested scheme" in response

    def test_fallback_to_default_template(self):
        chunks = [{"text": "Some info.", "source": "unknown.md", "distance": 1.0}]
        response = generate_response("unknown_intent", {}, chunks)
        assert "Some info." in response


class TestContract:
    def test_example_payload_valid(self):
        payload = example_asr_output().model_dump()
        validate_against_schema(payload, "asr_output")  # must not raise

    def test_invalid_payload_rejected(self):
        with pytest.raises(Exception):
            ASROutput(transcript="", detected_language="english!!", confidence_score=2.0)

    def test_nlu_query_roundtrip(self):
        q = NLUQuery(query="am I eligible for PM Kisan", language="en",
                     intents=["check_eligibility"], entities={"scheme_name": ["pm-kisan"]})
        assert q.query and q.intents == ["check_eligibility"]
