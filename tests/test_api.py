"""
API integration tests for GovConnect Edge Gateway endpoints.
"""

import io
import pytest
from fastapi.testclient import TestClient
from gateway.app import app
from gateway.schemas import Intent, Path
from services.tts.mock import MOCK_WAV_HEADER

client = TestClient(app)


def test_health_check():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "govconnect-edge"
    assert data["runtime_mode"] == "mock"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "path_distribution" in data


def test_web_demo_ui_accessible():
    response = client.get("/demo/", follow_redirects=True)
    assert response.status_code == 200
    assert "GovConnect Edge" in response.text



def test_text_query_cached_path():
    payload = {
        "session_id": "test_sess_001",
        "lang": "en",
        "text": "Who is eligible for PM KISAN scheme?",
    }
    response = client.post("/api/v1/text-query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["intent"]["name"] == Intent.CHECK_ELIGIBILITY.value
    assert data["slots"]["scheme_name"] == "pm_kisan"
    assert data["path"] == Path.CACHED.value
    assert "small and marginal farmer" in data["reply_text"].lower()
    assert data["audio"] is not None
    assert data["within_sla"] is True
    assert data["timings_ms"]["total"] >= 0.0


def test_text_query_out_of_scope():
    payload = {
        "session_id": "test_sess_002",
        "lang": "en",
        "text": "What is the weather forecast for tomorrow?",
    }
    response = client.post("/api/v1/text-query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["intent"]["name"] == Intent.OUT_OF_SCOPE.value
    assert data["path"] == Path.OFFLINE_EXTRACTIVE.value
    assert "only answer questions regarding" in data["reply_text"].lower()


def test_voice_query_end_to_end():
    # Use valid mock WAV bytes
    audio_stream = io.BytesIO(MOCK_WAV_HEADER)
    files = {"audio_file": ("test.wav", audio_stream, "audio/wav")}
    data = {
        "session_id": "voice_sess_001",
        "lang_hint": "hi",
    }
    response = client.post("/api/v1/voice-query", data=data, files=files)
    assert response.status_code == 200
    result = response.json()

    assert result["transcript"] != ""
    assert result["intent"]["name"] in [i.value for i in Intent]
    assert result["reply_text"] != ""
    assert result["audio"] is not None
    assert result["timings_ms"]["total"] > 0
