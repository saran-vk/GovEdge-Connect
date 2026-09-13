# 📋 Week 1 Requirements

### 1. Hardware & Environment Requirements
* **Development Workstations**: Multi-core CPU with CUDA-enabled GPU (NVIDIA RTX 4050 / RTX workstation GPU with 6GB+ VRAM) for local model inference, benchmarking, and embeddings.
* **Storage Environment**: Local persistent disk space for vector database storage (ChromaDB) and local model checkpoint caches (HuggingFace cache directory).
* **Execution Runtimes**: Python 3.10+ virtual environments (`venv` / `conda`) configured for speech processing and vector search dependencies.

---

### 2. Software, Libraries & Frameworks
* **NLU & Vector DB (Saran V)**:
  * **ChromaDB**: Local vector database for indexing and querying scheme documents.
  * **LangChain / LlamaIndex**: Frameworks for document processing, chunking, and vector embedding pipelines.
  * **Transformers & PyTorch**: For loading `IndicBERT` (`ai4bharat/indic-bert`) and sentence transformers (`sentence-transformers/all-MiniLM-L6-v2` or Indic embeddings).
  * **PyPDF2 / pdfplumber / Unstructured**: Tools for parsing and structuring raw e-Gov PDF documents and web FAQs.
* **Speech & Language AI (Sanjay Rathinam M N)**:
  * **OpenAI Whisper** (`openai-whisper`): Baseline ASR model environment.
  * **Torchaudio / HuggingFace Transformers**: For loading `Wav2Vec 2.0` / `IndicASR` model weights.
  * **JiWER**: Library for calculating Word Error Rate (WER) and Character Error Rate (CER).
  * **IndicTransv2 & AI4BTTS Dependencies**: Huggingface repositories / REST endpoints for neural machine translation and speech synthesis benchmarking.

---

### 3. Data & Knowledge Base Artifacts
* **e-Governance Document Corpus**: Official guidelines, eligibility criteria, required documents, and benefit details for major central/state welfare schemes (e.g., PM-KISAN, Ayushman Bharat, Rural Housing, State Welfare schemes).
* **Audio Datasets**: **AI4Bharat IndicVoices** corpora focusing on regional Dravidian/Indo-Aryan dialects (rural colloquial accents).
* **Intent & Entity Taxonomy**:
  * *Intents*: `check_eligibility`, `scheme_benefits`, `required_documents`, `application_status`, `general_inquiry`.
  * *Entities*: `scheme_name`, `age`, `income_level`, `occupation`, `caste_category`, `district/state`.

---

### 4. Technical Contracts & Interface Specs
* **ASR-to-RAG/NLU Interface Contract**: A standardized JSON schema mapping raw speech output to downstream processing:
  ```json
  {
    "transcript": "PM Kisan eligibility criteria for small farmers",
    "detected_language": "ta-IN",
    "confidence_score": 0.92,
    "timestamp": "2026-08-08T10:00:00Z"
  }
  ```

---

# 🗓️ Week 1 Execution Plan (02.08.2026 – 08.08.2026)

---

## Track A: NLU, RAG & Edge Systems
**Lead:** Saran V (Reg. No. 7376242AD294)

### Objective
Establish the foundational data pipeline, local vector store architecture, intent classification taxonomy, and interface boundaries for downstream query retrieval.

```
       ┌────────────────────────┐
       │ e-Gov Welfare Documents│
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Text Extraction &      │
       │ Recursive Chunking     │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Local ChromaDB Index   │
       └───────────┬────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  IndicBERT Intent Taxonomy Defined   │
│  (Eligibility / Benefits / Docs)     │
└──────────────────────────────────────┘
```

### Plan Breakdown
1. **Task A1: Document Corpus Collection & Structuring**
   * Gather official e-Gov welfare scheme documents, eligibility guidelines, and standard FAQs.
   * Parse semi-structured PDFs and web data into cleaned Markdown / plain text formats.
   * Define text chunking parameters (e.g., chunk size 512, overlap 64) optimized for retrieval precision.

2. **Task A2: Local ChromaDB Vector DB Setup**
   * Configure a local persistent ChromaDB workspace.
   * Set up baseline embedding models to vectorize scheme chunks and establish index schemas.

3. **Task A3: IndicBERT NLU Intent & Entity Taxonomy Definition**
   * Design intent taxonomy covering core citizen query categories (`eligibility`, `benefits`, `document_requirements`).
   * Draft entity annotation guidelines for regional user inputs.

4. **Task A4: Module Interface Specification**
   * Formulate the input JSON schema contract connecting ASR speech output to the RAG/NLU backend engine.

---

## Track B: Speech & Language AI
**Lead:** Sanjay Rathinam M N (Reg. No. 7376242AD288)

### Objective
Establish local environments for ASR, curate regional dialect speech audio, benchmark baseline speech recognition models (WER), and evaluate translation/TTS components.

```
       ┌────────────────────────┐
       │ AI4Bharat IndicVoices  │
       │ Rural Speech Datasets  │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Local Whisper &        │
       │ Wav2Vec 2.0 / IndicASR │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ WER Baseline Metrics   │
       │ Benchmarking           │
       └───────────┬────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ IndicTransv2 & AI4BTTS Profiling    │
└──────────────────────────────────────┘
```

### Plan Breakdown
1. **Task B1: Speech Dataset Curation**
   * Filter and extract regional speech subsets from AI4Bharat IndicVoices focusing on rural accents and colloquial variations.
   * Organize audio samples with matching ground-truth transcriptions.

2. **Task B2: ASR Local Environment Setup**
   * Set up local execution environments for OpenAI Whisper and Wav2Vec 2.0 / IndicASR.
   * Verify GPU execution and model inference scripts.

3. **Task B3: Baseline WER Benchmarking**
   * Run speech inference on test audio samples across regional dialects.
   * Measure baseline Word Error Rate (WER) to establish performance benchmarks for subsequent fine-tuning.

4. **Task B4: NMT & TTS Profiling**
   * Research integration requirements for `IndicTransv2` (translation layer).
   * Profile `AI4BTTS` to assess latency bottlenecks for real-time speech synthesis.

---

# 🎯 Week 1 Key Deliverables & Summary Matrix

| Module | Component | Week 1 Outcome / Deliverable |
| :--- | :--- | :--- |
| **RAG Pipeline** | ChromaDB Environment | Local ChromaDB vector database initialized with indexed scheme FAQs. |
| **NLU Engine** | IndicBERT Taxonomy | Defined intent classification classes (`eligibility`, `benefits`, `docs`). |
| **ASR Engine** | Whisper / IndicASR | Local environments configured; baseline WER benchmarks recorded. |
| **Speech Corpus** | IndicVoices | Rural accent audio subset extracted and formatted for training. |
| **Translation & TTS**| IndicTransv2 / AI4BTTS | Technical feasibility and latency profiling completed. |
| **System Architecture**| Module Interface Contract | Defined JSON payload contract connecting ASR output to RAG/NLU. |
