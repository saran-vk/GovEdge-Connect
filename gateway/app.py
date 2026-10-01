"""
FastAPI entrypoint and routing for GovConnect Edge orchestrator API.
"""

import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from gateway.schemas import (
    TextQueryReq,
    QueryRes,
    RuntimeMode,
)
from services.orchestrator import OrchestratorPipeline

app = FastAPI(
    title="GovConnect Edge Gateway",
    description="A Dialect-Adaptive, Voice-to-Voice Gateway for Inclusive e-Governance",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance configured via environment
DEFAULT_RUNTIME_MODE = RuntimeMode(os.getenv("RUNTIME_MODE", "mock").lower())
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.80"))
USE_NMT = os.getenv("USE_NMT", "true").lower() in ("true", "1", "yes")

orchestrator = OrchestratorPipeline(
    default_mode=DEFAULT_RUNTIME_MODE,
    conf_threshold=CONF_THRESHOLD,
    use_nmt=USE_NMT,
)

# Metric tracking in-memory registry
METRICS: Dict[str, Any] = {
    "total_queries": 0,
    "paths": {"cached": 0, "cloud_assisted": 0, "offline_extractive": 0},
    "total_latency_ms": 0.0,
}


@app.get("/healthz", tags=["System"])
async def health_check():
    """Liveness probe and system runtime mode check."""
    return {
        "status": "ok",
        "service": "govconnect-edge",
        "runtime_mode": DEFAULT_RUNTIME_MODE.value,
        "use_nmt": USE_NMT,
        "conf_threshold": CONF_THRESHOLD,
    }


@app.get("/metrics", tags=["System"])
async def get_metrics():
    """Returns aggregated per-stage and path performance metrics."""
    total = METRICS["total_queries"]
    avg_latency = (METRICS["total_latency_ms"] / total) if total > 0 else 0.0
    return {
        "total_queries": total,
        "path_distribution": METRICS["paths"],
        "avg_latency_ms": round(avg_latency, 2),
    }


@app.post("/api/v1/text-query", response_model=QueryRes, tags=["Gateway"])
async def handle_text_query(req: TextQueryReq) -> QueryRes:
    """Processes a text query and returns grounded text response and optional audio."""
    try:
        res = await orchestrator.process_query(
            session_id=req.session_id,
            text=req.text,
            lang=req.lang,
            runtime_mode=req.runtime_mode,
        )
        # Update metrics
        METRICS["total_queries"] += 1
        METRICS["paths"][res.path.value] = METRICS["paths"].get(res.path.value, 0) + 1
        METRICS["total_latency_ms"] += res.timings_ms.total
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing text query: {str(e)}",
        )


@app.post("/api/v1/voice-query", response_model=QueryRes, tags=["Gateway"])
async def handle_voice_query(
    session_id: str = Form(...),
    lang_hint: Optional[str] = Form(None),
    runtime_mode: Optional[str] = Form(None),
    audio_file: UploadFile = File(...),
) -> QueryRes:
    """Processes spoken audio (WAV 16kHz) through full Voice-to-Voice pipeline."""
    try:
        wav_bytes = await audio_file.read()
        if not wav_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio file provided.",
            )

        mode = RuntimeMode(runtime_mode) if runtime_mode else None
        res = await orchestrator.process_query(
            session_id=session_id,
            wav_bytes=wav_bytes,
            lang_hint=lang_hint,
            runtime_mode=mode,
        )
        # Update metrics
        METRICS["total_queries"] += 1
        METRICS["paths"][res.path.value] = METRICS["paths"].get(res.path.value, 0) + 1
        METRICS["total_latency_ms"] += res.timings_ms.total
        return res
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameter: {str(ve)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing voice query: {str(e)}",
        )
