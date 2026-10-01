# GovConnect Edge

> **A Dialect-Adaptive, Voice-to-Voice Gateway for Inclusive e-Governance**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Status](https://img.shields.io/badge/status-ready--for--execution-green.svg)]()

GovConnect Edge enables low-literacy rural citizens to query government welfare schemes (PM-KISAN, MGNREGA, Ayushman Bharat) in their native dialects over voice and receive grounded, accurate spoken responses without needing to navigate text portals or visit block administrative offices.

---

## 👥 Project Team

- **Sanjay Rathinam M. N.** — Speech & Language AI (ASR, IndicVoices preprocessing, LoRA fine-tuning, TTS)
- **Saran V** — NLU, Grounded RAG, Caching & Edge Systems (IndicBERTv2, ChromaDB, FastAPI orchestrator, Pi-profile evaluation)
- **Faculty Guide:** Prof. Poornima A, Department of AI & DS, Bannari Amman Institute of Technology

---

## 📖 Single Source of Truth

The complete prototype blueprint, scope locks, risk registers, and weekly execution gates are defined in:
👉 [GovConnect_Edge_Master_Plan.md](file:///home/saran/Projects/S5_mini_antiG/GovConnect_Edge_Master_Plan.md)

---

## 🏗️ System Architecture

```text
Voice In ──▶ Dialect ASR (faster-whisper / LoRA) ──▶ [Optional NMT: IndicTrans2]
                 │
                 ▼
         NLU (IndicBERTv2-SS) ── Intent & Scheme Slots
                 │
      ┌──────────┴──────────────────────────┐
      ▼ (High Confidence)                   ▼ (Low Confidence / Cache Miss)
Redis Cache (48 Pre-synthesized)     Grounded RAG (ChromaDB + bge-m3)
      │                                     │
      │                              Cloud LLM / Verbatim Extractive Fallback
      │                                     │
      └──────────────────┬──────────────────┘
                         ▼
             TTS (Bulbul / IndicTTS / Mock)
                         ▼
                     Voice Out
```

### Response Paths
1. **`cached`**: Resolved intent + scheme ($\ge 0.80$ confidence) $\rightarrow$ Pre-synthesized audio served from Redis ($< 1.5 - 2.0\text{s}$).
2. **`cloud_assisted`**: Cache miss $\rightarrow$ Grounded RAG + hosted cloud LLM with mandatory citations ($< 3.0\text{s}$).
3. **`offline_extractive`**: Local mode or network disconnect $\rightarrow$ Top RAG chunk read verbatim with `degraded: true` badge.

---

## ⚙️ Runtime Modes

The system operates under three switchable modes via `RUNTIME_MODE`:

| Mode | ASR | NMT | NLU | RAG | LLM | TTS | Use Case |
|---|---|---|---|---|---|---|---|
| `mock` | Canned responses | Passthrough | Rules | ChromaDB | Template | Silent/canned WAV | Rapid dev & decoupled team testing |
| `hybrid` | faster-whisper | IndicTrans2 | Fine-tuned IndicBERT | ChromaDB | Hosted LLM API | Bulbul / IndicTTS | Primary demo mode |
| `local` | CTranslate2 INT8 | Local | ONNX INT8 | ChromaDB | None (Extractive) | Cached audio only | Emulated Pi-profile benchmarking |

---

## 📁 Repository Layout

```text
S5_mini_antiG/
├── GovConnect_Edge_Master_Plan.md   # Master single source of truth
├── README.md                        # Project overview & quickstart
├── pyproject.toml                   # Project dependencies and packaging
├── .env.example                     # Environment configuration template
├── docker-compose.yml               # Redis, Gateway, and Pi-profile container definitions
├── Dockerfile                       # Container definition for gateway service
├── gateway/                         # FastAPI application, routing, and Pydantic schemas
│   ├── app.py
│   └── schemas.py
├── services/                        # Modular pipeline adapters
│   ├── audio_processor.py           # 16kHz audio normalization & validation
│   ├── asr/                         # Base, mock, and whisper adapters
│   ├── nmt/                         # Base, mock, and IndicTrans2 adapters
│   ├── nlu/                         # Base, rules, and IndicBERT adapters
│   ├── rag/                         # ChromaDB ingestion & filtered retriever
│   ├── responder/                   # Cache-first, grounded LLM, & extractive handlers
│   └── tts/                         # Base, mock, and Bulbul adapters
├── middleware/                      # Latency tracking and SLA evaluation
│   └── latency_tracker.py
├── data/                            # Datasets, scheme documents, and corpus metadata
│   ├── sources.md                   # Official source links & retrieval dates
│   ├── schemes/                     # Raw guideline PDFs / texts
│   ├── eval/                        # Graded evaluation queries
│   └── intents/                     # Intent training sets
├── eval/                            # Benchmarking & evaluation scripts
│   ├── wer.py                       # jiwer WER harness
│   ├── intent_eval.py               # Intent accuracy & confusion matrix
│   ├── hallucination_grading.py     # Faithfulness evaluation (<5% target)
│   └── latency_report.py            # Latency benchmarking by path
├── web-ui/                          # Browser demonstration interface
├── docs/                            # Documentation, contracts, and evaluation reports
│   ├── contract.md                  # Agreed interface contract signatures
│   ├── scope_lock.md                # Language tiers & inclusion gates
│   └── results/                     # Benchmark runs and logs
└── tests/                           # Contract tests for all adapters
    └── test_contracts.py
```

---

## 🚀 Getting Started

### 1. Setup Environment
```bash
# Clone/enter the repository
cd /home/saran/Projects/S5_mini_antiG

# Setup virtual environment with Python 3.11 using uv
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

### 2. Configure Environment
```bash
cp .env.example .env
```

### 3. Run Redis Service
```bash
docker compose up -d redis
```

### 4. Run Contract Tests
```bash
pytest tests/
```
