# Week 1 Action Items & Alignment Questionnaire

**Project:** GovConnect Edge  
**Addressed to:** Sanjay Rathinam M. N. & Prof. Poornima A (Guide)  
**From:** Saran V  
**Status:** In Progress (Week 1 Foundations)

---

## 1. Questions Requiring Sign-Off (Section 13)

| # | Item | Question / Proposal | Current Proposal | Sanjay / Guide Input |
|---|---|---|---|---|
| **Q1** | **GPU Plan** | Which compute resource will be used for Whisper-Small LoRA fine-tuning and how many hours? | Free Google Colab T4 / Kaggle P100 (estimated 15–20 GPU hours needed for Tier 1) | *Pending confirmation* |
| **Q2** | **TTS Engines** | Which TTS options are confirmed working for Tier 1 (Tamil, Hindi) and Tier 2 (Telugu, Malayalam)? | Bulbul API (Sarvam) / AI4Bharat IndicTTS / Coqui fallback | *Pending verification* |
| **Q3** | **LLM Policy** | Can the review prototype use hosted API endpoints (e.g. Sarvam API, Groq Llama 3.2), or must all generation run fully offline? | Hybrid mode with cloud API allowed for online demo; local extractive fallback demonstrated for edge compliance. | *Pending guide sign-off* |
| **Q4** | **Native Speaker Review** | Who on campus will verify Tamil, Hindi, and prospective Telugu/Malayalam queries and translated scheme chunks? | Team members (Saran - Tamil, Sanjay - Hindi/Tamil); lab peers for Telugu & Malayalam. | *To be finalized* |
| **Q5** | **Adapter Dates** | When will Sanjay's first real ASR (Whisper) and TTS adapters be delivered to plug into the interface contract? | Week 3 (Baseline ASR) & Week 4-5 (TTS integration). | *Agreed in timeline* |
| **Q6** | **Dataset Gate** | What is the minimum IndicVoices audio duration required per language to qualify for Tier 1 vs Tier 2? | Minimum 15–20 hours validated speech per language. | *To confirm in Week 1* |

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
