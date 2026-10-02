# Week 1 Action Items & Alignment Questionnaire

**Project:** GovConnect Edge  
**Addressed to:** Sanjay Rathinam M. N. & Prof. Poornima A (Guide)  
**From:** Saran V  
**Status:** In Progress (Week 1 Foundations)

---

## 1. Questions Requiring Sign-Off (Section 13)

| # | Item | Question / Proposal | Current Decision / Plan | Status |
|---|---|---|---|---|
| **Q1** | **GPU Plan** | Compute resource for Whisper-Small LoRA fine-tuning and hours? | Google Colab T4 GPU confirmed (reproducible LoRA + CTranslate2 script provided in `scripts/`). | **Confirmed** |
| **Q2** | **TTS Engines** | Confirmed TTS options for Tier 1 (Tamil, Hindi) & Tier 2 (Telugu, Malayalam)? | Sarvam Bulbul API + AI4Bharat IndicTTS both approved and integrated with graceful fallback. | **Confirmed** |
| **Q3** | **LLM Policy** | Can prototype use hosted API endpoints or must all generation run fully offline? | APIs allowed (Sarvam / Groq / OpenAI endpoints); extractive fallback retained for edge/offline mode. | **Confirmed** |
| **Q4** | **Native Speaker Review** | Who verifies Tamil, Hindi, Telugu, Malayalam queries and translated chunks? | Multi-dialect validation protocol established; colloquial datasets drafted in `data/eval/` and `data/intents/`. | **In Progress** |
| **Q5** | **Team Ownership & Scope** | Division between Speech AI (Sanjay) and RAG/Edge (Saran)? | **Unified Execution:** Both tracks merged under single pipeline. We build speech AI, ASR/TTS adapters, and RAG/Gateway together. | **Approved** |
| **Q6** | **Dataset Gate** | IndicVoices audio duration per language for Tier 1 qualification? | 15–20 hours validated speech per language on IndicVoices. | **Standardized** |

---

## 2. Completed Week 1 Milestones

- [x] Initialized Git repository and environment (.venv with Python 3.11 via `uv`).
- [x] Scaffolding for `gateway`, `services`, `middleware`, `data`, `eval`, `docs`, and `tests`.
- [x] `docker-compose.yml` for Redis and Pi-profile emulation.
- [x] Signed interface contract and data contracts ([docs/contract.md](file:///home/saran/Projects/S5_mini_antiG/docs/contract.md)).
- [x] End-to-end `mock` pipeline operational in [services/orchestrator.py](file:///home/saran/Projects/S5_mini_antiG/services/orchestrator.py).
- [x] Tested REST endpoints (`POST /api/v1/text-query`, `POST /api/v1/voice-query`, `GET /healthz`, `GET /metrics`).
- [x] Scheme guideline corpora created for PM-KISAN, MGNREGA, and Ayushman Bharat in [data/schemes/](file:///home/saran/Projects/S5_mini_antiG/data/schemes/).
- [x] 50 seed evaluation queries drafted and validated in [data/eval/50_seed_queries.json](file:///home/saran/Projects/S5_mini_antiG/data/eval/50_seed_queries.json).
- [x] 11 automated pytest tests passing.
