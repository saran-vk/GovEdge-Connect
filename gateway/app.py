"""
FastAPI entrypoint and routing for GovConnect Edge orchestrator API.
"""

import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

# Mount Web Demo UI
if os.path.exists("web-ui"):
    app.mount("/demo", StaticFiles(directory="web-ui", html=True), name="web-ui")

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/demo/")


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


from fastapi.responses import Response

@app.get("/api/v1/tts-stream", tags=["Gateway"])
async def stream_tts(text: str, lang: str = "ta"):
    """Streams synthesized MP3 audio for given text and language."""
    from services.tts.speech_engine import generate_speech
    audio_bytes = generate_speech(text, lang)
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.post("/api/v1/tts", tags=["Gateway"])
async def generate_tts_endpoint(payload: Dict[str, str]):
    """Returns base64 synthesized audio for text and language."""
    text = payload.get("text", "")
    lang = payload.get("lang", "ta")
    from services.tts.speech_engine import get_cached_speech_b64
    audio_b64 = get_cached_speech_b64(text, lang)
    return {"audio": audio_b64, "lang": lang}


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


@app.websocket("/api/v1/voice-stream")
async def websocket_voice_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice query streaming with progressive stage telemetry.
    Accepts JSON handshake or audio chunk packets, returns real-time pipeline events and audio.
    """
    await websocket.accept()
    session_id = "ws_session_default"
    lang_hint = "hi"
    runtime_mode = None
    audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()
            if "text" in message and message["text"]:
                import json
                try:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type", "")

                    if msg_type == "handshake":
                        session_id = payload.get("session_id", session_id)
                        lang_hint = payload.get("lang_hint", lang_hint)
                        if "runtime_mode" in payload and payload["runtime_mode"]:
                            runtime_mode = RuntimeMode(payload["runtime_mode"])
                        await websocket.send_json({"event": "ready", "session_id": session_id})

                    elif msg_type == "finish":
                        # Process buffered audio
                        await websocket.send_json({"event": "stage", "stage": "processing"})
                        mode = runtime_mode or DEFAULT_RUNTIME_MODE
                        res = await orchestrator.process_query(
                            session_id=session_id,
                            wav_bytes=bytes(audio_buffer) if audio_buffer else None,
                            lang_hint=lang_hint,
                            runtime_mode=mode,
                        )

                        # Update metrics
                        METRICS["total_queries"] += 1
                        METRICS["paths"][res.path.value] = METRICS["paths"].get(res.path.value, 0) + 1
                        METRICS["total_latency_ms"] += res.timings_ms.total

                        # Send pipeline events and final result
                        await websocket.send_json({
                            "event": "complete",
                            "response": res.model_dump(),
                        })
                        audio_buffer.clear()

                except Exception as ex:
                    await websocket.send_json({"event": "error", "detail": str(ex)})

            elif "bytes" in message and message["bytes"]:
                audio_buffer.extend(message["bytes"])
                await websocket.send_json({
                    "event": "buffered",
                    "bytes_received": len(message["bytes"]),
                    "total_buffer_size": len(audio_buffer),
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"event": "error", "detail": str(e)})
        except Exception:
            pass

