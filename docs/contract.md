# Interface & Data Contract (Signed Specification)

**Project:** GovConnect Edge  
**Participants:** Sanjay Rathinam M. N. (Speech & Language AI) & Saran V (NLU, RAG & Edge Systems)  
**Status:** Agreed Baseline (Week 1)

---

## 1. Adapter Interface Contracts

Every adapter (ASR, NMT, NLU, TTS) must adhere strictly to these asynchronous signatures.  
**Rule:** Every adapter MUST return execution duration in milliseconds (`ms`), allowing latency tracking middleware to calculate accurate per-stage timings without external timing overhead or guesswork.

### 1.1 ASR (Automatic Speech Recognition)
- **Signature:** `transcribe(wav_bytes: bytes, lang_hint: Optional[str] = None) -> ASRResult`
- **Output:**
  ```json
  {
    "text": "pm kisan yojana eligibility",
    "lang": "ta",
    "confidence": 0.94,
    "ms": 420.5
  }
  ```

### 1.2 NMT (Neural Machine Translation)
- **Signature:** `translate(text: str, src_lang: str, tgt_lang: str) -> NMTResult`
- **Output:**
  ```json
  {
    "text": "How do I check eligibility for PM KISAN?",
    "ms": 115.2
  }
  ```

### 1.3 NLU (Natural Language Understanding)
- **Signature:** `extract(text: str, lang: str = "en") -> NLUResult`
- **Output:**
  ```json
  {
    "intent": {
      "name": "check_eligibility",
      "confidence": 0.92
    },
    "slots": {
      "scheme_name": "pm_kisan",
      "state": null
    },
    "ms": 85.0
  }
  ```

### 1.4 TTS (Text-to-Speech)
- **Signature:** `synthesize(text: str, lang: str = "hi") -> TTSResult`
- **Output:**
  ```json
  {
    "audio_wav_bytes": "<raw 16kHz mono WAV bytes>",
    "ms": 610.8
  }
  ```

---

## 2. Shared Data Contracts & Schemas

### 2.1 Intents (`Intent` Enum)
Strictly locked to 4 core scheme intents + 1 out-of-scope refusal:
1. `check_eligibility`
2. `documents_needed`
3. `apply_process`
4. `benefit_amount`
5. `out_of_scope`

*Note:* Scheme is a **slot**, not an intent.

### 2.2 Execution Paths (`Path` Enum)
1. `cached`: Resolved intent + scheme with confidence ≥ `CONF_THRESHOLD` (0.80). Answer audio and text retrieved directly from Redis cache.
2. `cloud_assisted`: Cache miss; retrieved RAG context sent to cloud LLM endpoint; audio synthesized live.
3. `offline_extractive`: Cloud unreachable or local/offline mode; verbatim reading of top RAG chunk with `degraded: true`.

### 2.3 Cache Key Formula
$$\text{key} = \text{sha256}(\text{scheme} + \text{intent} + \text{lang} + \text{corpus\_version})$$
Size: 3 schemes × 4 intents × 4 languages = 48 pre-synthesized answers.

---

## 3. SLA Targets

| Execution Path | Target End-to-End Latency | Profile |
|---|---|---|
| `cached` | < 1.5 – 2.0 s | Pi-profile (emulated CPU limit) |
| `cloud_assisted` | < 3.0 s | Server / Hybrid mode |
| `offline_extractive` | < 1.5 s | Pi-profile / Degraded mode |
