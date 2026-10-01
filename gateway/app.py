"""
FastAPI entrypoint for GovConnect Edge orchestrator API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="GovConnect Edge Gateway",
    description="Voice-to-Voice Gateway for Inclusive e-Governance",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", tags=["System"])
async def health_check():
    return {"status": "ok", "service": "govconnect-edge"}
