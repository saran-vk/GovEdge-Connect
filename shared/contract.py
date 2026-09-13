"""A4 - Module interface contract between ASR output and the RAG/NLU backend.

Implements the JSON payload contract from the Week 1 plan:

    {
      "transcript": "...",
      "detected_language": "ta-IN",
      "confidence_score": 0.92,
      "timestamp": "2026-08-08T10:00:00Z"
    }

ASROutput  - raw speech -> text (produced by Track B ASR).
NLUQuery   - normalized query -> downstream RAG (produced by Track A NLU).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from shared.config import REPO_ROOT, resolve

_LANG_RE = re.compile(r"^[a-z]{2,3}(-[A-Z]{2,3})?$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ASROutput(BaseModel):
    """Standardized ASR payload handed to the NLU/RAG layer."""

    transcript: str = Field(..., min_length=1, description="Raw ASR text output")
    detected_language: str = Field(..., description="BCP-47 tag, e.g. ta-IN")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    timestamp: str = Field(default_factory=_utc_now, description="ISO-8601 UTC")

    @field_validator("detected_language")
    @classmethod
    def _validate_lang(cls, v: str) -> str:
        if not _LANG_RE.match(v):
            raise ValueError(f"invalid BCP-47 language tag: {v!r}")
        return v


class NLUQuery(BaseModel):
    """Normalized downstream query produced from an ASROutput."""

    query: str
    language: str
    intents: list[str] = Field(default_factory=list)
    entities: dict[str, list[str]] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)


def load_contract_schema() -> dict:
    schema_path = REPO_ROOT / "config" / "asr_nlu_contract.json"
    with open(schema_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_against_schema(instance: dict, schema_kind: str = "asr_output") -> None:
    """Validate a payload against the contract JSON schema using jsonschema."""
    from jsonschema import ValidationError, validate  # noqa: PLC0415

    schema = load_contract_schema().get(schema_kind)
    if schema is None:
        raise ValueError(f"unknown contract kind: {schema_kind}")

    try:
        validate(instance=instance, schema=schema)
    except ValidationError as exc:
        raise ValueError(f"contract validation failed: {exc.message}") from exc


def example_asr_output() -> ASROutput:
    return ASROutput(
        transcript="PM Kisan eligibility criteria for small farmers",
        detected_language="ta-IN",
        confidence_score=0.92,
        timestamp="2026-08-08T10:00:00Z",
    )


def validate_contract_sample() -> dict:
    """Validates the contract doc's sample payload - used in smoke tests."""
    payload = example_asr_output().model_dump()
    validate_against_schema(payload, "asr_output")
    return payload
