# Week 3 & 4 — Execution Plan

_Created: September 2026 | GovConnect Edge Project_
_References: `S5_Mini/plans/S5_PLAN.md`, `docs/week2_problem_formulation.md`_

## Overview

Weeks 3–4 assume Week 2 is completed. Allocation follows a re-balanced split:

- **Week 2** = core implementation (ASR GPU upgrade, corpus expansion, retrieval
  optimization, classifier training, language detection).
- **Week 3** = finish any Week 2 overflow **plus** the analysis/design items from
  the S5_PLAN (vector DB review, fine-tuning exploration, NMT, speech pipeline
  design, FastAPI services, TTS integration).
- **Week 4** = mathematical evaluation framework, quantization/memory fit, full
  hardware characterization, final E2E demo, docs, and delivery.

## Week 3 — Optimization, Integration & Deep Analysis

### Track A — NLU/RAG & Edge (Lead: Saran V)

| # | Task | Detail | Deliverable |
|:--|:--|:--|:--|
| A1 | Corpus expansion (W2 overflow) | Ingest real e-Gov PDFs into `track_a/data/raw/`; reach 20+ sheets / 100+ chunks | Extracted + processed corpus |
| A2 | Metadata-filtered retrieval (W2 overflow) | Add scheme_name + language filters in `vector_store.py`; build precision@4 harness | Filtered RAG + precision metrics |
| A3 | Classifier training (W2 overflow) | Label 100+ multilingual queries; train IndicBERT head; target >80% accuracy | Trained classifier + eval report |
| A4 | Vector DB lit review & design | ChromaDB vs FAISS (metadata filtering, latency, 6GB VRAM budget) | `docs/week3_vector_db_review.md` |
| A5 | Response improvement | Scheme-aware formatting, confidence gating, multilingual templates (en/hi/ta) | Updated responder + templates |

### Track B — Speech & Language AI (Lead: Sanjay Rathinam M N)

| # | Task | Detail | Deliverable |
|:--|:--|:--|:--|
| B1 | ASR upgrade (W2 overflow) | Whisper large-v3 on RTX 4050 CUDA int8; targets Tamil <25% / Hindi <15% WER, sub-1s ASR | Updated ASR + WER report |
| B2 | Fine-tuning + augmentation | LoRA/QLoRA on IndicVoices if large-v3 insufficient; noise/telephony simulation for >70% accent adaptability | Fine-tune notes + augmentation set |
| B3 | Language detection | Pre-ASR language ID step (en/hi/ta) | Language detector + integration |
| B4 | IndicTransv2 NMT | Obtain HF token; replace NLLB-200 fallback; multi-dialect query routing | NMT integration + latency profile |
| B5 | Speech pipeline design | Document ASR → NMT → TTS architecture + augmentation strategy | `docs/week3_speech_design.md` |

### Integration (Joint)

| # | Task | Detail | Deliverable |
|:--|:--|:--|:--|
| I1 | TTS into E2E pipeline | Wire mms-tts into response flow (audio out) | Voice output in E2E demo |
| I2 | FastAPI microservices (start) | Local containerized services: ASR, NLU/RAG, TTS, ChromaDB-backed | Service layer + API |
| I3 | Latency harness | Per-stage + E2E timings on Ryzen 7 + RTX 4050; contract validation on every hop | Latency report |

## Week 4 — Evaluation, Finalization & Delivery

| # | Task | Detail | Deliverable |
|:--|:--|:--|:--|
| E1 | Mathematical eval framework | WER/CER `(S+D+I)/N`, cosine-similarity retrieval math, chunking math (512/64), precision@4 | `docs/week4_evaluation.md` |
| E2 | Quantization & memory fit | INT8/F16 (~60% compression); verify all models fit 6GB VRAM | Quantized model config |
| E3 | Hardware characterization | Sub-1s ASR, **<3s E2E** latency verification on RTX 4050 | Benchmark report |
| E4 | FastAPI microservices (complete) | Containerized, local ChromaDB, API endpoint for responses | Complete API demo |
| E5 | Final E2E demo | Audio → Lang-detect → ASR → Contract → NLU → RAG → Response → TTS (validated) | Recorded/final demo |
| E6 | Docs & deliverables | `docs/week4_final_report.md`, updated README, Second Review presentation (extends `GovConnect_First_Review.pptx`) | Final report + slides |
| E7 | Git hygiene | Commit all W1–W4 work cleanly on `govedge_working` | Clean repo state |

## Weekly Cadence

1. **Sunday** — sprint planning; pick tasks from the matrices above.
2. **Tue/Thu** — sync via the frozen Interface Contract (`config/asr_nlu_contract.json`).
3. **Saturday** — run `scripts/run_all.sh`, regenerate conclusions, commit.

## Definition of Done (project-level)

| Metric | Target |
|:--|:--|
| Tamil WER | <25% |
| Hindi WER | <15% |
| Intent accuracy | >80% |
| Retrieval precision@4 | >70% |
| End-to-end latency | <3s |
| Hallucinated responses | 0 |
| Cloud dependency | None (fully offline) |