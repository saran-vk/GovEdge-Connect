"""
GovConnect Edge Pipeline Orchestrator.
Glues together Audio Processing -> ASR -> NMT -> NLU -> Cache/RAG/LLM -> TTS.
"""

import base64
import os
from typing import Optional

from gateway.schemas import (
    Intent,
    IntentResult,
    Path,
    QueryRes,
    RuntimeMode,
    Slots,
)
from middleware.latency_tracker import LatencyTracker
from services.asr.base import BaseASRAdapter
from services.asr.mock import MockASRAdapter
from services.asr.whisper import WhisperASRAdapter
from services.audio_processor import AudioProcessor
from services.nlu.base import BaseNLUAdapter
from services.nlu.indicbert import IndicBERTNLUAdapter
from services.nlu.rules import RulesNLUAdapter
from services.nmt.base import BaseNMTAdapter
from services.nmt.indictrans2 import IndicTrans2NMTAdapter
from services.nmt.mock import MockNMTAdapter
from services.rag.retriever import RAGRetriever
from services.responder.cache import ResponseCache
from services.responder.extractive import ExtractiveResponder
from services.responder.llm import GroundedLLMResponder
from services.tts.base import BaseTTSAdapter
from services.tts.bulbul import BulbulTTSAdapter
from services.tts.mock import MockTTSAdapter

OUT_OF_SCOPE_RESPONSES = {
    "en": "I can only answer questions regarding PM-KISAN, MGNREGA, and Ayushman Bharat.",
    "hi": "मैं केवल पीएम-किसान, मनरेगा और आयुष्मान भारत योजनाओं से संबंधित प्रश्नों का उत्तर दे सकता हूँ।",
    "ta": "என்னால் பிஎம்-கிசான், மகாத்மா காந்தி ஊரக வேலைவாய்ப்பு மற்றும் ஆயுஷ்மான் பாரத் திட்டங்கள் குறித்த கேள்விகளுக்கு மட்டுமே பதிலளிக்க முடியும்.",
    "te": "నేను కేవలం PM-KISAN, MGNREGA మరియు ఆయుష్మాన్ భారత్ పథకాల గురించిన ప్రశ్నలకు మాత్రమే సమాధానం ఇవ్వగలను.",
    "ml": "എനിക്ക് പിഎം-കിസാൻ, തൊഴിലുറപ്പ് പദ്ധതി (MGNREGA), ആയുഷ്മാൻ ഭാരത് എന്നീ പദ്ധതികളെക്കുറിച്ചുള്ള ചോദ്യങ്ങൾക്ക് மட்டுமே മറുപടി നൽകാൻ കഴിയൂ.",
}


class OrchestratorPipeline:
    """End-to-end pipeline orchestrator for voice and text query flows."""

    def __init__(
        self,
        default_mode: RuntimeMode = RuntimeMode.MOCK,
        conf_threshold: float = 0.80,
        use_nmt: bool = True,
    ):
        self.default_mode = default_mode
        self.conf_threshold = conf_threshold
        self.use_nmt = use_nmt

        # Shared services
        self.audio_processor = AudioProcessor()
        self.cache = ResponseCache()
        self.retriever = RAGRetriever()
        self.extractive_responder = ExtractiveResponder()
        self.llm_responder = GroundedLLMResponder()

        # Adapters per mode
        self.mock_asr = MockASRAdapter()
        self.real_asr = WhisperASRAdapter()

        self.mock_nmt = MockNMTAdapter()
        self.real_nmt = IndicTrans2NMTAdapter()

        self.mock_nlu = RulesNLUAdapter()
        self.real_nlu = IndicBERTNLUAdapter()

        self.mock_tts = MockTTSAdapter()
        self.real_tts = BulbulTTSAdapter()

    def _get_adapters(self, mode: RuntimeMode):
        if mode == RuntimeMode.MOCK:
            return self.mock_asr, self.mock_nmt, self.mock_nlu, self.mock_tts
        elif mode == RuntimeMode.HYBRID:
            return self.real_asr, self.real_nmt, self.real_nlu, self.real_tts
        else:  # LOCAL (INT8 / Quantized)
            return self.real_asr, self.mock_nmt, self.real_nlu, self.mock_tts

    async def process_query(
        self,
        session_id: str,
        text: Optional[str] = None,
        wav_bytes: Optional[bytes] = None,
        lang: str = "en",
        lang_hint: Optional[str] = None,
        runtime_mode: Optional[RuntimeMode] = None,
    ) -> QueryRes:
        """Process incoming query through the complete GovConnect pipeline."""
        tracker = LatencyTracker()
        mode = runtime_mode or self.default_mode
        asr_adapter, nmt_adapter, nlu_adapter, tts_adapter = self._get_adapters(mode)

        transcript = ""
        current_lang = lang_hint or lang

        # Stage 1: Audio Processing & ASR (if voice input)
        if wav_bytes is not None:
            clean_wav, _ = self.audio_processor.process_wav(wav_bytes)
            asr_res = await asr_adapter.transcribe(clean_wav, lang_hint=current_lang)
            tracker.record_stage("asr", asr_res["ms"])
            transcript = asr_res["text"]
            current_lang = asr_res.get("lang", current_lang)
            query_text = transcript
        else:
            query_text = text or ""
            transcript = query_text

        # Stage 2: NMT (Optional hop if language != en and USE_NMT enabled)
        normalized_text = query_text
        if self.use_nmt and current_lang != "en":
            nmt_res = await nmt_adapter.translate(query_text, src_lang=current_lang, tgt_lang="en")
            tracker.record_stage("nmt", nmt_res["ms"])
            normalized_text = nmt_res["text"]

        # Stage 3: NLU Intent & Slot Extraction
        nlu_res = await nlu_adapter.extract(normalized_text, lang=current_lang)
        tracker.record_stage("nlu", nlu_res["ms"])
        intent_res: IntentResult = nlu_res["intent"]
        slots: Slots = nlu_res["slots"]

        # Stage 4: Path Decision & Response Generation
        path = Path.CLOUD_ASSISTED
        degraded = False
        citations = []
        reply_text = ""
        audio_b64 = None

        # Case A: Out-of-Scope Intent -> Refuse safely without LLM generation
        if intent_res.name == Intent.OUT_OF_SCOPE:
            reply_text = OUT_OF_SCOPE_RESPONSES.get(current_lang, OUT_OF_SCOPE_RESPONSES["en"])
            path = Path.OFFLINE_EXTRACTIVE
            tts_res = await tts_adapter.synthesize(reply_text, lang=current_lang)
            tracker.record_stage("tts", tts_res["ms"])
            audio_b64 = base64.b64encode(tts_res["audio_wav_bytes"]).decode("ascii")

        # Case B: Confident Scheme + Intent -> Cache-first lookup
        elif intent_res.confidence >= self.conf_threshold and slots.scheme_name:
            cached_text, cached_audio, cache_ms = await self.cache.get(
                scheme=slots.scheme_name,
                intent=intent_res.name.value,
                lang=current_lang,
            )
            tracker.record_stage("llm_or_cache", cache_ms)

            if cached_text:
                path = Path.CACHED
                reply_text = cached_text
                audio_b64 = cached_audio
            else:
                # Cache miss -> Fallback to RAG + LLM/Extractive
                reply_text, audio_b64, citations, path, degraded = await self._generate_uncached(
                    query_text=normalized_text,
                    scheme_name=slots.scheme_name,
                    lang=current_lang,
                    mode=mode,
                    tts_adapter=tts_adapter,
                    tracker=tracker,
                )

        # Case C: Ambiguous intent or scheme -> RAG + LLM/Extractive
        else:
            reply_text, audio_b64, citations, path, degraded = await self._generate_uncached(
                query_text=normalized_text,
                scheme_name=slots.scheme_name,
                lang=current_lang,
                mode=mode,
                tts_adapter=tts_adapter,
                tracker=tracker,
            )

        # Finalize latency & SLA status
        timings_ms, within_sla = tracker.finalize(path)

        return QueryRes(
            transcript=transcript,
            normalized_text=normalized_text,
            intent=intent_res,
            slots=slots,
            citations=citations,
            reply_text=reply_text,
            audio=audio_b64,
            path=path,
            degraded=degraded,
            timings_ms=timings_ms,
            within_sla=within_sla,
        )

    async def _generate_uncached(
        self,
        query_text: str,
        scheme_name: Optional[str],
        lang: str,
        mode: RuntimeMode,
        tts_adapter: BaseTTSAdapter,
        tracker: LatencyTracker,
    ):
        # 1. RAG Retrieval
        chunks, citations, rag_ms = await self.retriever.retrieve(query_text, scheme_name=scheme_name)
        tracker.record_stage("rag", rag_ms)

        # 2. Responder selection (Extractive if local or offline fallback)
        if mode == RuntimeMode.LOCAL:
            path = Path.OFFLINE_EXTRACTIVE
            reply_text, citations, ext_ms = self.extractive_responder.extract_reply(chunks)
            tracker.record_stage("llm_or_cache", ext_ms)
            degraded = True
        else:
            path = Path.CLOUD_ASSISTED
            reply_text, citations, llm_ms = await self.llm_responder.generate_response(
                query_text, chunks, lang=lang
            )
            tracker.record_stage("llm_or_cache", llm_ms)
            degraded = False

        # 3. Live TTS Synthesis
        tts_res = await tts_adapter.synthesize(reply_text, lang=lang)
        tracker.record_stage("tts", tts_res["ms"])
        audio_b64 = base64.b64encode(tts_res["audio_wav_bytes"]).decode("ascii")

        return reply_text, audio_b64, citations, path, degraded
