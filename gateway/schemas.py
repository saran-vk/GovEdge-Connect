"""
Data contracts for GovConnect Edge as defined in GovConnect_Edge_Master_Plan.md Section 6.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class RuntimeMode(str, Enum):
    MOCK = "mock"
    HYBRID = "hybrid"
    LOCAL = "local"


class Intent(str, Enum):
    CHECK_ELIGIBILITY = "check_eligibility"
    DOCUMENTS_NEEDED = "documents_needed"
    APPLY_PROCESS = "apply_process"
    BENEFIT_AMOUNT = "benefit_amount"
    OUT_OF_SCOPE = "out_of_scope"


class Path(str, Enum):
    CACHED = "cached"
    CLOUD_ASSISTED = "cloud_assisted"
    OFFLINE_EXTRACTIVE = "offline_extractive"


class IntentResult(BaseModel):
    name: Intent
    confidence: float = Field(ge=0.0, le=1.0)


class Slots(BaseModel):
    scheme_name: Optional[str] = None
    state: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    scheme: str
    chunk_id: str
    score: float


class TimingsMs(BaseModel):
    asr: float = 0.0
    nmt: float = 0.0
    nlu: float = 0.0
    rag: float = 0.0
    llm_or_cache: float = 0.0
    tts: float = 0.0
    total: float = 0.0


class SchemeChunk(BaseModel):
    id: str
    text: str
    scheme_name: str
    eligibility: Optional[str] = None
    state: Optional[str] = None
    source_doc: str
    page: Optional[int] = None
    lang: str = "en"
    corpus_version: str = "1.0"


class TextQueryReq(BaseModel):
    session_id: str
    lang: str = "en"
    text: str
    runtime_mode: Optional[RuntimeMode] = None


class VoiceQueryReq(BaseModel):
    session_id: str
    lang_hint: Optional[str] = None
    runtime_mode: Optional[RuntimeMode] = None


class QueryRes(BaseModel):
    transcript: str = ""
    normalized_text: str = ""
    intent: IntentResult
    slots: Slots = Field(default_factory=Slots)
    citations: List[Citation] = Field(default_factory=list)
    reply_text: str
    audio: Optional[str] = None  # URL or base64 audio
    path: Path
    degraded: bool = False
    timings_ms: TimingsMs = Field(default_factory=TimingsMs)
    within_sla: bool = True
