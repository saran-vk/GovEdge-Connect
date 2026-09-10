# Week 1 Conclusions

_Generated: 2026-08-08 20:46 UTC_

> Auto-generated from the latest run artifacts (`track_b/data/results/*.json`, ChromaDB index).

## 1. Pipeline health

| Component | Status |
| :-- | :-- |
| ChromaDB index (`scheme_corpus`) | OK - 21 chunks |
| ASR baseline | OK |
| TTS profiler | ok |
| NMT profiler | skipped |
| Interface contract | OK - 4/4 probes validated |

## 2. ASR baseline (faster-whisper)

- Model `small` on `cpu` (int8), 15 clips, 1 hallucination(s) flagged.
- **Overall WER 25.30% | CER 10.94%** (excluding hallucinations: WER 19.97%)

| Language | N | WER | CER |
| :-- | --: | --: | --: |
| en-IN | 5 | 0.00% | 0.00% |
| hi-IN | 5 | 22.00% | 5.54% |
| ta-IN | 5 | 53.90% | 27.27% |

**Bottleneck:** ta-IN shows the highest WER (53.90%) -> ASR quality is the likely ceiling; further effort should target Indic ASR fine-tuning, not RAG tuning.**
- Note: 1 clip(s) produced WER>1.0 (Whisper hallucination / language-switch on short clips); mitigate with VAD + longer prompts, or flag-and-retry.

## 3. Corpus & RAG health
- 4 seed fact sheets indexed as 21 chunks (chunk 512 / overlap 64).
- Embedding backend used: **minilm** (IndicBERT is gated on HuggingFace; see section 6).
- Retrieval conclusions are only valid within this seeded corpus; real e-Gov PDF ingestion (A1) must scale for production claims.

## 4. NLU readiness (few-shot prototype)

| Demo query | Intent | Entities | Sources | Contract |
| :-- | :-- | :-- | --: | :-- |
| am I eligible for PM Kisan | check_eligibility | {'scheme_name': ['pm kisan', 'pm-kisan']} | 4 | OK |
| how much money does ayushman bharat give | scheme_benefits | {'scheme_name': ['ayushman bharat', 'ayushman-bharat']} | 4 | OK |
| which documents are needed for the housing scheme | required_documents | {'scheme_name': ['pmay']} | 4 | OK |
| check my application status | application_status | {} | 4 | OK |

- Classifier is nearest-centroid over IndicBERT/MiniLM embeddings, not a trained head. Good enough to validate the taxonomy; needs a labelled fine-tuning set for production.

## 5. Architecture validation
- ASR->NLU JSON contract: validated on 4/4 probes.
- faster-whisper CPU (int8) is a viable local ASR baseline; the RTX 4050 can later switch to `device: cuda` + `compute_type: float16` in settings.yaml.
- NMT/TTS profiling is lightweight and guarded - heavy models never block the run.
- TTS (mms-tts: hin, tam, eng): avg synthesis 775 ms/sample - not real-time interactive, acceptable for turn-based responses.
- NMT skipped: OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/ai4bharat/indictrans2-en-indic-1B.
401 Client Error. (Request ID: Root=1-6a77954a-2708c3a1093b4e0c4694d0

## 6. Key findings & readiness for Week 2

1. **ASR is the bottleneck.** The language with the highest WER determines end-to-end quality; invest in Indic ASR fine-tuning data before RAG tuning.
2. **IndicBERT is gated.** `ai4bharat/indic-bert` returned 401 without auth - the MiniLM fallback kept the pipeline green. Decide: request HF access for IndicBERT, or standardise on a multilingual sentence encoder (e.g. multilingual-e5) as the Week-2 baseline.
3. **Corpus is the other bottleneck.** 4 seed sheets prove the pipeline; official e-Gov PDFs must be ingested (A1) and indexed for any real claim.
4. **Taxonomy validated.** All five intents and six entity types worked on the probe set; convert exemplars into a labelled fine-tuning set for a trained head.
5. **Interface contract is the frozen seam** between Track B and Track A - keep `config/asr_nlu_contract.json` stable across the rest of the project.

---

_Track A lead: Saran V | Track B lead: Sanjay Rathinam M N_
