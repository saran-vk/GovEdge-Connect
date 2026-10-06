"""
Contract tests verifying that mock adapters conform to the agreed interface specifications.
"""

import pytest
from services.asr.mock import MockASRAdapter
from services.nmt.mock import MockNMTAdapter
from services.nlu.rules import RulesNLUAdapter
from services.tts.mock import MockTTSAdapter
from gateway.schemas import Intent, Path, QueryRes, IntentResult, Slots, TimingsMs


@pytest.mark.asyncio
async def test_asr_adapter_contract():
    adapter = MockASRAdapter()
    dummy_wav = b"RIFF....WAVE"
    res = await adapter.transcribe(dummy_wav, lang_hint="ta")

    assert "text" in res and isinstance(res["text"], str)
    assert "lang" in res and isinstance(res["lang"], str)
    assert "confidence" in res and isinstance(res["confidence"], float)
    assert "ms" in res and isinstance(res["ms"], float)
    assert res["ms"] >= 0


@pytest.mark.asyncio
async def test_nmt_adapter_contract():
    adapter = MockNMTAdapter()
    res = await adapter.translate("வணக்கம்", src_lang="ta", tgt_lang="en")

    assert "text" in res and isinstance(res["text"], str)
    assert "ms" in res and isinstance(res["ms"], float)
    assert res["ms"] >= 0


@pytest.mark.asyncio
async def test_nlu_adapter_contract():
    adapter = RulesNLUAdapter()
    res = await adapter.extract("How do I apply for PM KISAN?")

    assert "intent" in res and isinstance(res["intent"], IntentResult)
    assert res["intent"].name in [i.value for i in Intent]
    assert 0.0 <= res["intent"].confidence <= 1.0
    assert "slots" in res and isinstance(res["slots"], Slots)
    assert "ms" in res and isinstance(res["ms"], float)


@pytest.mark.asyncio
async def test_tts_adapter_contract():
    adapter = MockTTSAdapter()
    res = await adapter.synthesize("PM KISAN eligibility details", lang="hi")

    assert "audio_wav_bytes" in res and isinstance(res["audio_wav_bytes"], bytes)
    assert "ms" in res and isinstance(res["ms"], float)
    assert len(res["audio_wav_bytes"]) > 0


def test_query_res_schema():
    res = QueryRes(
        transcript="pm kisan",
        normalized_text="pm kisan",
        intent=IntentResult(name=Intent.CHECK_ELIGIBILITY, confidence=0.9),
        reply_text="Eligible farmers get Rs 6000/year.",
        path=Path.CACHED,
        timings_ms=TimingsMs(total=120.0),
    )
    assert res.intent.name == Intent.CHECK_ELIGIBILITY
    assert res.path == Path.CACHED
    assert res.timings_ms.total == 120.0


@pytest.mark.asyncio
async def test_real_adapters_contracts():
    from services.asr.whisper import WhisperASRAdapter
    from services.nmt.indictrans2 import IndicTrans2NMTAdapter
    from services.nlu.indicbert import IndicBERTNLUAdapter
    from services.tts.bulbul import BulbulTTSAdapter
    from services.tts.mock import MOCK_WAV_HEADER

    # Real ASR Adapter Contract
    asr = WhisperASRAdapter()
    asr_res = await asr.transcribe(MOCK_WAV_HEADER, lang_hint="ta")
    assert "text" in asr_res and isinstance(asr_res["text"], str)
    assert "lang" in asr_res and isinstance(asr_res["lang"], str)
    assert "confidence" in asr_res and isinstance(asr_res["confidence"], float)
    assert "ms" in asr_res and asr_res["ms"] >= 0

    # Real NMT Adapter Contract
    nmt = IndicTrans2NMTAdapter()
    nmt_res = await nmt.translate("பிஎம் கிசான் தகுதி", src_lang="ta", tgt_lang="en")
    assert "text" in nmt_res and isinstance(nmt_res["text"], str)
    assert "ms" in nmt_res and nmt_res["ms"] >= 0

    # Real NLU Adapter Contract
    nlu = IndicBERTNLUAdapter()
    nlu_res = await nlu.extract("Who is eligible for PM KISAN?")
    assert "intent" in nlu_res and isinstance(nlu_res["intent"], IntentResult)
    assert nlu_res["intent"].name in [i.value for i in Intent]
    assert 0.0 <= nlu_res["intent"].confidence <= 1.0
    assert "slots" in nlu_res and isinstance(nlu_res["slots"], Slots)
    assert "ms" in nlu_res and nlu_res["ms"] >= 0

    # Real TTS Adapter Contract
    tts = BulbulTTSAdapter()
    tts_res = await tts.synthesize("PM KISAN eligibility details", lang="hi")
    assert "audio_wav_bytes" in tts_res and isinstance(tts_res["audio_wav_bytes"], bytes)
    assert "ms" in tts_res and tts_res["ms"] >= 0
    assert len(tts_res["audio_wav_bytes"]) > 0
