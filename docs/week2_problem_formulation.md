# Week 2 — Problem Formulation

_Created: September 2026 | GovConnect Edge Project_

## 1. Speech Engineering Problem

### Problem Statement
Design an offline-capable multilingual ASR system that achieves <25% WER for Tamil
and <15% WER for Hindi on consumer hardware (RTX 4050, 6GB VRAM), supporting
code-mixed queries common in rural Indian speech.

### Current Baseline (Week 1 Results)
| Metric | Value | Target |
|:--|:--|:--|
| Tamil WER | 38.00% | <25% |
| Hindi WER | 33.50% | <15% |
| English WER | 0.00% | <5% |
| Overall WER | 23.83% | <15% |
| Hallucination clips | 0 | 0 |
| ASR latency (CPU) | ~10s | <1s (GPU) |

### Approach
1. **Model selection**: Test Whisper large-v3 (1.5B params, fits in 6GB VRAM with int8)
2. **Initial prompt optimization**: Tune the `initial_prompt` parameter with domain-
   specific vocabulary to improve language detection and transcription accuracy
3. **Fine-tuning exploration**: Investigate LoRA/QLoRA fine-tuning of Whisper on
   IndicVoices data for Tamil-specific improvement
4. **GPU acceleration**: Enable CUDA inference on RTX 4050 for sub-1s latency

## 2. NLU/RAG Pipeline Problem

### Problem Statement
Design a grounded RAG system that retrieves relevant government scheme information
with >70% precision@4 and generates zero-hallucination responses across 5 intent
types in 3 languages, operating entirely offline.

### Current Baseline (Week 1 Results)
| Metric | Value | Target |
|:--|:--|:--|
| Corpus size | 8 sheets / 46 chunks | 20+ sheets / 100+ chunks |
| Intent classification | Nearest-centroid (untrained) | >80% accuracy |
| Retrieval | Top-4 L2 distance | >70% precision@4 |
| Response quality | Template-based | Structured + contextual |
| Languages | 3 (en/hi/ta) | 3+ (expandable) |

### Approach
1. **Corpus expansion**: Ingest real e-Gov PDFs from official sources
2. **Retrieval optimization**: Add metadata filtering (by scheme_name, language)
3. **Classifier evaluation**: Create labeled test set (100+ queries), measure accuracy
4. **Response improvement**: Add scheme-name-aware formatting in templates

## 3. Integration Problem

### Problem Statement
Design a seamless ASR → NLU → RAG → Response pipeline that processes a voice query
and returns a relevant answer in <3 seconds end-to-end on local hardware.

### Current Pipeline Flow
```
Audio → ASR (faster-whisper) → JSON Contract → NLU (intent + entity)
  → RAG (ChromaDB top-k) → Response (template) → TTS (optional)
```

### Target Pipeline Flow
```
Audio → Language Detection → ASR (CUDA) → Contract Validation
  → NLU (intent + entity + confidence check) → RAG (metadata-filtered)
  → Response (structured) → TTS (optional) → Audio Output
```

### Key Technical Contracts
- ASR output must match `config/asr_nlu_contract.json` schema
- NLU query must pass jsonschema validation before RAG
- Response must include source attribution for transparency
- All components must operate offline (no cloud API calls)
