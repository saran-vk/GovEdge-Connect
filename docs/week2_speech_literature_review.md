# Week 2 — Speech Processing Literature Review

_Conducted: September 2026 | Track B lead: Sanjay Rathinam M N_

## 1. AI4Bharat IndicVoices

AI4Bharat (Indian Institute of Technology Madras) has developed IndicVoices, one of the
largest multilingual speech datasets for Indian languages. Key findings:

- **Scale**: 7,000+ hours of speech data across 22 Indic languages, including low-resource
  Dravidian languages (Tamil, Telugu, Kannada, Malayalam).
- **Dialect coverage**: Includes rural colloquial accents and phonetic deviations that
  standard ASR models struggle with — exactly the population our welfare assistant targets.
- **Recording conditions**: Mix of studio and telephony-quality audio, relevant to our
  IVR/phone-based deployment scenario.
- **Gap identified**: IndicVoices is primarily a training dataset, not a pre-trained model.
  It requires fine-tuning of base models (Whisper, Wav2Vec2) on this data, which demands
  significant GPU compute beyond our 6GB VRAM budget.
- **Relevance to GovConnect**: Could be used as fine-tuning data for Whisper to improve
  Tamil WER (currently 38% with medium model). However, the compute cost requires
  prioritization against other deliverables.

## 2. IndicASR

IndicASR is AI4Bharat's pre-trained ASR model family built on Wav2Vec 2.0 and XLS-R
architectures:

- **Models**: Pre-trained on 22 Indian languages using IndicVoices data.
- **Performance**: Achieves ~10-15% WER on clean Indic speech, but degrades significantly
  on noisy/telephony audio and dialectal variations.
- **Architecture**: Wav2Vec 2.0 (self-supervised pre-training) + CTC head for
  transcription. Lighter than Whisper but less multilingual.
- **Gap identified**: IndicASR models are not available via faster-whisper (CTranslate2).
  They require PyTorch inference, which is slower on CPU and harder to optimize for edge
  deployment.
- **Comparison with our Whisper medium**: Whisper medium (769M params) is larger but
  handles code-mixing better. IndicASR is smaller but requires language-specific heads.
- **Recommendation**: Keep Whisper as primary ASR; consider IndicASR as an alternative
  for specific language-only deployments where code-mixing is not expected.

## 3. Bhashini

Bhashini is the Government of India's national language technology platform:

- **Architecture**: Cloud-based API providing ASR, NMT, TTS, and OCR across 22+ Indian
  languages.
- **Strengths**: High accuracy (trained on massive government data), official government
  backing, integration with UMANG and other e-Governance platforms.
- **Critical gaps for GovConnect**:
  - **Cloud dependency**: Requires internet connectivity — contradicts our offline-first
    edge deployment requirement.
  - **Cost**: API usage incurs per-request costs, not suitable for free public service.
  - **Latency**: Network round-trip adds 200-500ms per request, exceeding our sub-1s
    target.
  - **Feature-phone exclusion**: Bhashini targets smartphone users; our IVR/telephony
    use case is not supported.
- **Relevance**: Bhashini validates the government's push toward multilingual AI, but
  our offline-first approach addresses the gap Bhashini cannot fill.

## 4. Sarvam AI

Sarvam AI is an Indian AI startup building Indic language AI models:

- **Products**: Sarvam-2B (Indic LLM), Sarvam-ASR (speech recognition), Sarvam-TTS
  (speech synthesis).
- **Strengths**: Purpose-built for Indian languages; open-weight models available on
  HuggingFace; competitive with larger multilingual models on Indic benchmarks.
- **Gap identified**: Sarvam models are cloud-optimized. The 2B parameter LLM requires
  ~4GB VRAM in float16, fitting our RTX 4050 but not edge devices like Raspberry Pi.
- **Relevance**: Sarvam-ASR could be an alternative to Whisper for Indic-specific ASR.
  Sarvam-2B could replace our template-based responder with a proper LLM for response
  generation (Week 3+ consideration).

## 5. Problem Formulation — Speech Engineering

Based on the literature review, the core speech engineering problem for GovConnect is:

**Primary**: Achieve <20% WER for Tamil on local hardware (RTX 4050, 6GB VRAM) using
offline ASR, given that:
- Whisper medium achieves 38% WER on Tamil (current baseline)
- Whisper large-v3 may achieve 20-25% but requires ~6GB VRAM in int8
- IndicASR achieves 10-15% on clean audio but degrades on telephony quality
- Fine-tuning on IndicVoices could improve by 5-10% but requires compute investment

**Secondary**: Optimize the multi-stage voice processing pipeline (ASR → optional NMT →
response → TTS) to achieve:
- Sub-1.0s ASR latency on GPU (current: ~10s on CPU with medium model)
- Sub-1.5s total pipeline latency for turn-based responses
- Support for code-mixed queries (Hinglish, Tanglish) that are common in rural India

## 6. Our Positioning vs. Existing Solutions

| Dimension | Bhashini | Sarvam AI | GovConnect (Ours) |
|:--|:--|:--|:--|
| Deployment | Cloud API | Cloud / API | **Offline edge** |
| Cost | Per-request | Per-request | **Free (local compute)** |
| Feature-phone | No | No | **Yes (IVR-ready)** |
| Languages | 22+ | 22+ | 3 (en/hi/ta) — expandable |
| Accuracy | High | High | Medium (improving) |
| Privacy | Data leaves device | Data leaves device | **Data stays local** |

Our competitive advantage is the offline-first, zero-cost, privacy-preserving design
that serves citizens excluded by cloud-dependent solutions.
