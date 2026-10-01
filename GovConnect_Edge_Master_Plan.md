# GovConnect Edge — Master Prototype Plan (single source of truth)

**Project:** GovConnect Edge: A Dialect-Adaptive, Voice-to-Voice Gateway for Inclusive e-Governance
**Team:** Sanjay Rathinam M. N. (Speech & Language AI) · Saran V (NLU, RAG & Edge Systems)
**Guide:** Prof. Poornima A, Dept. of AI & DS, Bannari Amman Institute of Technology
**Basis:** v1.0 plan + First Review PPTX + Revised Execution Plan, merged with the best parts of `oc_idea.md` and `agy_idea.md`, and hardened by the pre-mortem.
**Status:** Plan only. No code written yet. Anything marked **VERIFY** must be confirmed in Week 1.

---

## 0. How to read this document

- Sections 1–3: what we are building and what we promise (scope, locked decisions, targets).
- Sections 4–7: how it is built (architecture, runtime modes, data contracts, RAG).
- Sections 8–9: how we prove it works (evaluation, edge strategy).
- Sections 10–12: when and who (timeline, risks, early warning signs).
- Sections 13–14: what is still open and what to do in the first 7 days.

If this file conflicts with `oc_idea.md`, `agy_idea.md`, or v1.0, **this file wins**. If it conflicts with the Revised Execution Plan, the Revised Plan wins unless a deviation is listed in Section 2.

---

## 1. Mission and scope

**Mission.** A voice-first, multilingual gateway so low-literacy rural citizens can ask about welfare schemes in their own language over a phone or browser, and hear a grounded answer back, without text portals or a trip to the block office.

**Schemes (seed corpus):** PM-KISAN, MGNREGA, Ayushman Bharat.

**Pipeline (unchanged from all three submitted documents):**

```
Voice In → Dialect ASR → Translation (IndicTrans2) → NLU Intent (IndicBERTv2-SS)
        → Grounded RAG (ChromaDB) → Cached template OR LLM answer → TTS → Voice Out
```

**In scope for the prototype**
- Browser-based voice-to-voice demo with full per-stage latency and evidence display.
- Four languages, delivered in tiers (Section 2, decision D2).
- Cache-first response layer, cloud-assisted generative path, and extractive offline fallback.
- Measured evaluation: WER per language, intent accuracy, hallucination rate, per-stage latency.
- Quantized ASR + NLU (INT8), benchmarked on the development machine under a CPU-limited "Pi profile".

**Out of scope for the prototype (stretch only)**
- Real Raspberry Pi 4 hardware measurements (no device available).
- Live telephony (Twilio/Asterisk). A webhook stub exists, wired to nothing.
- More than four languages.
- Fine-tuning IndicTrans2.
- On-device generative LLM.

---

## 2. Locked decisions

| # | Decision | Reason |
|---|---|---|
| D1 | **Demo surface is a browser web app.** Pi 4 and IVR are not required for the review. | Confirmed by Saran; no Pi available. |
| D2 | **Languages are tiered.** Tier 1: **Tamil, Hindi** (full pipeline). Tier 2: **Telugu, Malayalam** (added only if they pass the gates in Section 7.3). | Revised Plan says lock 2–3 languages from measured baseline WER. Four is the stretch, not the promise. |
| D3 | **Stack:** Python 3.11, FastAPI + Uvicorn, ChromaDB, Redis, Docker Compose, faster-whisper / CTranslate2, ONNX Runtime. | Matches all submitted documents. |
| D4 | **Three runtime modes via one flag: `RUNTIME_MODE=mock | hybrid | local`.** Every ASR, NMT, LLM and TTS call goes through an adapter with a mock and a real implementation. | Compute (GPU) and API access (Bulbul/Sarvam/IndicVoices) are not guaranteed. This decouples the two team members. |
| D5 | **Intents:** `check_eligibility`, `documents_needed`, `apply_process`, `benefit_amount`, plus `out_of_scope`. Scheme is a **slot**, not an intent. | Matches the submitted documents. `scheme_overview` / `general_faq` from `agy_idea` are dropped. |
| D6 | **Cache-first is the architecture, not an optimization.** Common (scheme × intent × language) answers are pre-synthesized. | Revised Plan, Phase 2–4. |
| D7 | **IndicTrans2 is pretrained only**, with a Week-6 ablation to test whether it helps (flag `USE_NMT`, default on). | Revised Plan. The ablation is new: IndicBERTv2 is multilingual and may not need the NMT hop. |
| D8 | **Quantize ASR + NLU only.** The LLM stays server/cloud hosted. | Revised Plan. |
| D9 | **Hallucination is measured:** target <5% on a held-out set of ≥50 graded queries. No "0%" claim anywhere. | Revised Plan. |
| D10 | **Edge numbers are labeled "Pi-profile (emulated)"** and never presented as measured Pi 4 results. | Honesty. A desktop CPU under a limit is faster than a Cortex-A72. |

---

## 3. Targets (what we promise at review)

| Metric | Target | How measured |
|---|---|---|
| Dialect ASR quality | >70% accuracy (i.e. WER ≤ 30%) on Tier 1 languages after LoRA, with before/after table vs. the Week-3 baseline. Tier 2 reported honestly even if below target. | `jiwer` on held-out IndicVoices subset |
| Intent accuracy | ≥ 85% on a held-out intent test set (Tier 1) | Confusion matrix |
| Hallucination rate | <5% on ≥50 held-out real scheme queries, manually graded | Graded eval set (Section 8) |
| Online end-to-end latency | <3.0 s (server/hybrid mode, text→audio path including cache misses where the LLM is a hosted API) | Per-stage timers, 20–30 query run |
| Cached-path latency | <1.5–2.0 s under Pi-profile (emulated) | Same harness, CPU-limited container |
| Novel-query latency | Reported separately as "cloud-assisted," with its own number | Same harness |
| Offline behavior | Documented and demonstrated: top RAG chunk read aloud with a `degraded` flag | Demo scenario |
| Languages demonstrated | 2 guaranteed (Tier 1), 4 target | Section 7.3 gates |

The intent target (85%) and the 70% ASR figure are team-set goals. If the baseline shows they are unreachable for a language, report the real number and move the language to Tier 2. Do not tune the target to fit the result.

---

## 4. Architecture

```
[Browser demo UI]
   │  POST /api/v1/text-query
   │  POST /api/v1/voice-query        (wav, 16 kHz mono)
   │  WS   /api/v1/voice-stream       (chunked TTS back)
   ▼
[orchestrator-api  (FastAPI)]  ── per-stage timer + path decision ──────────────┐
   ├─▶ audio_processor     resample 16 kHz mono, trim silence, validate
   ├─▶ asr_adapter         mock | faster-whisper (base/small, LoRA later) | remote API
   ├─▶ nmt_adapter         mock | IndicTrans2 pretrained (flag USE_NMT)
   ├─▶ nlu_service         rules stub → fine-tuned IndicBERTv2-SS (4 intents + out_of_scope, + slots)
   ├─▶ rag_service         ChromaDB + bge-m3, metadata filter by scheme_name + intent
   ├─▶ responder
   │      ├─ cache hit?    Redis (scheme, intent, lang, corpus_version) → pre-synthesized audio
   │      ├─ cloud LLM     Sarvam-2B / Llama-3.2-3B, RAG-grounded, citation-enforced
   │      └─ extractive    top RAG chunk verbatim (offline / degraded)
   └─▶ tts_adapter         mock | Bulbul / IndicTTS | fallback engine
[eval-harness]  WER · intent accuracy · hallucination grading · latency report
[metrics]       /metrics  (per-stage timings), /healthz
```

**Execution paths (returned in every response as `path`):**
1. `cached`: intent + scheme confidently resolved, answer audio served from Redis. No LLM call.
2. `cloud_assisted`: cache miss, RAG chunks sent to the LLM endpoint, TTS synthesized live.
3. `offline_extractive`: cloud unreachable and not cached; top RAG chunk is read verbatim, `degraded: true`.

**Decision rules**
- Cache is used only when NLU confidence ≥ threshold (default 0.80) **and** the scheme slot is resolved. Otherwise go to RAG.
- If intent is `out_of_scope` or retrieval score is below threshold, respond with a safe "I can only answer about PM-KISAN, MGNREGA and Ayushman Bharat" message in the user's language. Do not generate.
- The LLM prompt must forbid answering outside retrieved context and must return the chunk IDs it used. Responses without citations are rejected and fall back to extractive.

---

## 5. Runtime modes

| Mode | ASR | NMT | NLU | RAG | LLM | TTS | Use |
|---|---|---|---|---|---|---|---|
| `mock` | canned transcripts by audio hash | passthrough | rules | real ChromaDB | template | pre-recorded / silent clip | Fast dev, CI, and developing Saran's half without Sanjay's models |
| `hybrid` | faster-whisper local | IndicTrans2 local or API | fine-tuned local | real | hosted API (Sarvam/Groq/etc.) | Bulbul API or local | Main demo mode |
| `local` | faster-whisper INT8 | local | ONNX INT8 | real | none (cache + extractive only) | cached audio only | Pi-profile (emulated) benchmarking |

Rule: **a stage is not "done" until it has both a mock and a real adapter that pass the same contract test.** This is the fix for the biggest pre-mortem blind spot (the Saran/Sanjay stub gap).

**Interface contract between the two halves (agree in Week 1):**
- `ASR: (wav_bytes, lang_hint) → {text, lang, confidence, ms}`
- `TTS: (text, lang) → {audio_wav_bytes, ms}`
- `NMT: (text, src_lang, tgt_lang) → {text, ms}`
All adapters return `ms` so the timer middleware never guesses.

---

## 6. Data contracts

```python
class Intent(str, Enum):
    CHECK_ELIGIBILITY = "check_eligibility"
    DOCUMENTS_NEEDED  = "documents_needed"
    APPLY_PROCESS     = "apply_process"
    BENEFIT_AMOUNT    = "benefit_amount"
    OUT_OF_SCOPE      = "out_of_scope"

class Path(str, Enum):
    CACHED = "cached"
    CLOUD_ASSISTED = "cloud_assisted"
    OFFLINE_EXTRACTIVE = "offline_extractive"

class VoiceQueryReq:   session_id, lang_hint?, audio_wav (16k mono), runtime_mode?
class TextQueryReq:    session_id, lang, text, runtime_mode?

class SchemeChunk:     id, text, scheme_name, eligibility, state, source_doc, page, lang, corpus_version

class QueryRes:
    transcript, normalized_text,
    intent {name, confidence}, slots {scheme_name?, state?},
    citations [{scheme, chunk_id, score}],
    reply_text, audio (url or base64),
    path: Path, degraded: bool,
    timings_ms {asr, nmt, nlu, rag, llm_or_cache, tts, total},
    within_sla: bool
```

**Cache key:** `sha256(scheme + intent + lang + corpus_version)`. Invalidate by bumping `corpus_version` when the scheme documents change.

**Cache size sanity check:** 3 schemes × 4 intents × 4 languages = 48 pre-synthesized answers. Small enough to build and verify by hand.

---

## 7. Data, languages and RAG

### 7.1 Scheme corpus (Saran)
- Collect official guideline PDFs/portal pages for PM-KISAN, MGNREGA, Ayushman Bharat. Record source URL and retrieval date in a `sources.md`.
- Chunk with `RecursiveCharacterTextSplitter` (~500 / 50). Embed with `BAAI/bge-m3`. Metadata: `scheme_name, eligibility, state, source_doc, page, lang`.
- Answers in Tier 1/2 languages: store the **English/Hindi source chunk**, and handle language at the generation/translation boundary. Do not machine-translate the whole corpus silently; any translated chunk is flagged `translated: true` and reviewed by a native speaker.
- Retrieval test: for each scheme, 10 hand-written queries per language must return the correct chunk in the top 3 before RAG is called "done".

### 7.2 Speech data (Sanjay)
- Filter IndicVoices for the four languages; standardize to 16 kHz mono WAV; keep a fixed train/validation/test split from the start (no leakage between fine-tune and benchmark).
- Baselines per language: `whisper-base`, plus `IndicASR`-family model (**VERIFY** exact model name and availability).

### 7.3 Language gates (decide Tier 2 inclusion at end of Week 3)
A language is promoted to Tier 1 or kept in Tier 2 only if it passes **all** of:
1. **Data gate:** enough IndicVoices audio to train/validate (number agreed in Week 1).
2. **ASR gate:** a baseline WER exists, and fine-tune is plausible in the compute budget.
3. **TTS gate:** a working TTS voice for that language is confirmed (**VERIFY** for Telugu and Malayalam on Bulbul/IndicTTS before assuming).
4. **Eval gate:** ≥ 50 evaluation queries written and answer-checked by a fluent speaker.

Fail a gate → that language ships as "text-in/text-out demo only" or is dropped. The review report states which.

---

## 8. Evaluation plan (built in Phase 1, not at the end)

| Eval | Set | Owner | Output |
|---|---|---|---|
| WER | Held-out IndicVoices test split, per language | Sanjay | Baseline vs. each fine-tune iteration table |
| Intent accuracy | ≥ 200 labeled queries (Tier 1), written in dialect-plausible wording | Saran | Accuracy + confusion matrix |
| Retrieval hit-rate | 10 queries × 3 schemes × language | Saran | Top-1 / top-3 hit rate |
| **Hallucination / faithfulness** | **≥ 50 held-out real scheme queries**, manually graded against the source documents | Joint | Rate, with each failure categorized |
| Latency | 20–30 representative queries, end-to-end, per stage | Joint | Table by path (cached / cloud-assisted / extractive) |
| Cascade error | Same queries as text vs. as speech | Joint | Shows how much ASR error costs downstream |

**Grading rubric for hallucination:** an answer fails if it states any fact (amount, document, eligibility rule, deadline) not supported by a cited chunk. Out-of-scope refusals count as correct.

Query-writing rule: seed queries in English, rewrite colloquially in each language with a native speaker, and **freeze the set before the first fine-tune run**.

---

## 9. Edge strategy (no Pi required)

- **Pi-profile (emulated):** run the `local` mode in Docker with CPU limits approximating a Pi 4 (4 cores, constrained memory, no GPU). Report results as *emulated*.
- **INT8 quantization:** ASR through CTranslate2/faster-whisper INT8; NLU through ONNX Runtime INT8. Report model size, memory, and latency before vs. after, on the same machine.
- **Why this is honest:** emulated numbers are a lower bound on real Pi latency. The report states this and lists real-Pi validation as future work. If a Pi can be borrowed (department lab, friend), run the same harness once and add the real column.
- **Offline behavior** is demonstrated by stopping the LLM endpoint during the demo and showing the extractive fallback with the `degraded` badge.

---

## 10. Timeline (12 weeks) with exit gates

| Weeks | Phase | Saran | Sanjay | Exit gate |
|---|---|---|---|---|
| **1** | Foundations | Repo + Docker Compose skeleton; adapter interfaces + mock mode; intent schema; start corpus collection | Confirm IndicVoices access; confirm TTS options per language; GPU plan (Colab/Kaggle/lab) | Interface contract signed; `mock` mode returns a full response end to end; every **VERIFY** item answered |
| **2–3** | Data + baselines | ChromaDB built; retrieval tests; write ≥ 50 eval queries + intent set | Preprocess audio; baseline WER per language | **Scope lock:** Tier 1/Tier 2 decided against the Section 7.3 gates; documented |
| **4–5** | Core AI | Fine-tune IndicBERTv2-SS on intents; intent-filtered retrieval; responder + cache builder | LoRA fine-tune Whisper-Small (Tier 1); IndicTrans2 integration; TTS adapter; cache audio generation | Each stage has mock + real adapter passing the same contract tests |
| **6** | **Checkpoint 1** | Integration | Integration | 10–15 queries through ASR→(NMT)→NLU→RAG→TTS; NMT ablation result; integration bugs logged and triaged |
| **7** | Core AI close | Tier 2 languages added if they passed gates; hallucination guard (citation enforcement) | WER re-benchmark vs. baseline | WER table with before/after; intent accuracy reported |
| **8–9** | Backend + UI | FastAPI endpoints, WebSocket streaming, browser demo UI, `/metrics` | Streaming TTS chunks; per-stage timing in adapters | Browser demo works end to end in `hybrid` mode |
| **9 (end)** | **Checkpoint 2** | Latency run | Latency run | 20–30 queries with per-stage timings; top latency/correctness issues fixed |
| **10–11** | Edge + quantization | INT8 NLU (ONNX), Pi-profile benchmark harness, offline fallback | INT8 ASR (CTranslate2); optional Twilio/Asterisk webhook sandbox if time allows | Cached path measured under Pi-profile; offline fallback demonstrated |
| **12** | Evaluation + buffer | Final hallucination grading, report | Final WER/latency tables, report | **Last 2–3 days are unscheduled buffer.** Demo is rehearsed, not first-run. |

Slippage rule: if a gate is missed by more than 3 days, cut scope (Tier 2 languages first, then telephony stub, then streaming), never the evaluation work or buffer.

---

## 11. Risk register (merged: Revised Plan + pre-mortem)

| Risk | Impact | Mitigation | Trigger to act |
|---|---|---|---|
| Whisper-Small weak on Tamil/Telugu/Malayalam colloquial speech | Garbage transcripts break NLU | Per-language baseline first; LoRA on priority languages; compare IndicASR-family models; honest reporting | Baseline WER > ~50% for a language → Tier 2 / text-only |
| No GPU for LoRA | Fine-tuning never happens | Decide in Week 1: Colab/Kaggle/lab quota; small LoRA on Whisper-Small; keep fine-tune scope to Tier 1 | No compute plan by end of Week 1 |
| TTS unavailable for a language | "Voice-to-voice" becomes voice-to-text for it | Gate in Section 7.3; fallback engine; label clearly in demo | Any language without confirmed TTS by Week 3 |
| API access (Bulbul/Sarvam/IndicVoices) delayed | Real mode blocked | `mock` / `hybrid` modes decouple development | Access still pending Week 2 |
| Cross-team stub gap | Saran's half cannot be demonstrated alone | Interface contract + mock adapters in Week 1 | No signed contract by end of Week 1 |
| Four languages promised before evidence | Shallow, unverified results | Tiering and gates | Any slide says "4 languages" before the scope lock |
| ASR → NMT → NLU error stacking | Hidden accuracy loss | Cascade-error eval (text vs. speech); NMT ablation in Week 6 | Text-input accuracy ≫ speech-input accuracy |
| Eval set has no owner | Hallucination claim indefensible | Owned by Saran, frozen before first fine-tune | < 50 graded queries at Week 6 |
| Hallucination in LLM path | Wrong policy information | Citation-enforced prompt; reject uncited answers; out-of-scope refusal; <5% measured target | Any graded failure involving amount/eligibility |
| Integration bugs found late | No time to fix | Checkpoints in Weeks 6 and 9; 2–3 day buffer | Checkpoint slips |
| No Pi hardware | Edge claim unsupported | Pi-profile emulation labeled honestly; borrow a Pi if possible | Anyone writes an emulated number as "measured on Pi" |
| Policy documents change | Stale answers | `corpus_version` + source/date log | Source page differs from indexed text |

---

## 12. Early warning signs (check at day 30)

- [ ] Baseline WER table exists for all four languages (end of Week 3).
- [ ] TTS availability confirmed per language, in writing.
- [ ] Signed ASR/TTS/NMT interface contract; `mock` mode works end to end.
- [ ] At least one real non-English retrieval test has passed.
- [ ] A named GPU plan exists (where and how many hours).
- [ ] The 50-query evaluation set exists and is frozen.
- [ ] The intent list is still exactly four plus `out_of_scope`.

Any unchecked box at day 30 is a scope-cut meeting, not a "we'll catch up".

---

## 13. Open items (answer these to unblock the build)

1. **GPU plan:** which resource will run LoRA (Colab, Kaggle, college lab)? How many hours?
2. **TTS access:** which engine and which languages do we actually have (Bulbul, IndicTTS, other)?
3. **LLM policy:** may the prototype use a hosted LLM API key, or must it run fully local/stubbed for college compliance?
4. **Native-speaker review:** who checks Tamil, Telugu, Malayalam and Hindi evaluation queries and translated chunks?
5. **Sanjay's schedule:** agree dates for the interface contract and for the first working ASR/TTS adapters.
6. **Dataset size gate:** minimum IndicVoices hours per language to qualify for Tier 1.

---

## 14. First 7 days

1. Send the open items in Section 13 to Sanjay and the guide; get answers on GPU, TTS and LLM policy.
2. Create the repo skeleton below with Docker Compose and the `RUNTIME_MODE` switch.
3. Write the adapter interfaces and the mock implementations; make `mock` mode return a full `QueryRes`.
4. Collect the three scheme documents and start `sources.md`.
5. Draft the 50 evaluation queries in English (translations come later).
6. Agree and sign the ASR/TTS/NMT contract with Sanjay.

**Suggested repo layout**

```
govconnect-edge/
  docker-compose.yml
  .env.example                 # RUNTIME_MODE, USE_NMT, CONF_THRESHOLD, LLM endpoint
  gateway/                     # FastAPI app, routes, WebSocket, IVR stub
  services/
    audio_processor.py
    asr/      {base.py, mock.py, whisper.py}
    nmt/      {base.py, mock.py, indictrans2.py}
    nlu/      {base.py, rules.py, indicbert.py}
    rag/      {ingest.py, retriever.py}
    responder/{cache.py, llm.py, extractive.py}
    tts/      {base.py, mock.py, bulbul.py}
  middleware/latency_tracker.py
  data/       {schemes/, sources.md, eval/, intents/}
  eval/       {wer.py, intent_eval.py, hallucination_grading.py, latency_report.py}
  web-ui/     # recorder, transcript, intent, citations, audio, timing, path badge
  docs/       {contract.md, scope_lock.md, results/}
  tests/      # contract tests: every adapter's mock and real pass the same suite
```

---

## 15. Definition of done (prototype review)

- Browser demo takes spoken Tier 1 queries and returns spoken answers, showing transcript, intent, citations, path badge and per-stage timings.
- WER table (baseline → fine-tuned) per language, with honest Tier 2 results.
- Intent accuracy and a graded hallucination rate on ≥ 50 held-out queries, reported as measured numbers.
- Latency table split by path: cached (Pi-profile, emulated), cloud-assisted, offline-extractive.
- Offline fallback live-demonstrated.
- A written list of what was **not** done (real Pi, live telephony, languages beyond the tier reached).
