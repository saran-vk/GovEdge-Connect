# Week 2 — Conversational e-Governance Literature Review

_Conducted: September 2026 | Track A lead: Saran V_

## 1. Jugalbandi

Jugalbandi is an open-source conversational AI platform for Indian government services,
built by AI4Bharat and deployed via WhatsApp:

- **Architecture**: User sends voice/text message on WhatsApp → ASR transcription →
  LLM-based intent understanding → API call to government service → response generation
  → TTS synthesis → reply on WhatsApp.
- **Strengths**: Real-world deployment with 100,000+ users; supports 10+ Indian languages;
  integrates with actual government APIs (PF, PAN, Aadhaar).
- **Critical gaps for GovConnect**:
  - **WhatsApp dependency**: Excludes feature-phone users and areas with limited
    smartphone penetration — exactly the rural population we target.
  - **Cloud-only**: All processing happens on cloud servers; no offline capability.
  - **LLM hallucination**: Uses large language models for intent understanding and
    response generation, which can hallucinate on complex government policy questions.
    This is why our project uses a grounded RAG approach instead.
  - **Cost**: Cloud compute costs scale with usage; not sustainable for free public
    service at national scale.
- **Relevance**: Jugalbandi validates the conversational e-Governance use case but
  highlights the need for offline, grounded, zero-hallucination alternatives.

## 2. Bhashini (e-Governance Layer)

Beyond the speech processing platform (covered in speech review), Bhashini has an
e-Governance integration layer:

- **UMANG integration**: Bhashini powers multilingual voice interfaces for the UMANG
  (Unified Mobile Application for New-age Governance) platform.
- **Strengths**: Government backing, official API integrations, 22-language support.
- **Gaps**: Same cloud dependency and cost issues as the base platform. Additionally,
  UMANG integration requires smartphone apps, not accessible via IVR or feature phones.
- **Relevance**: Validates that government is investing in multilingual AI for governance,
  but the delivery mechanism (smartphone app) excludes our target population.

## 3. Sarvam AI (e-Governance Applications)

Sarvam AI has piloted e-Governance applications of their Indic language models:

- **Use cases**: Document digitization, voice-based form filling, multilingual customer
  service for government portals.
- **Strengths**: Purpose-built for Indian languages; strong performance on Indic benchmarks.
- **Gaps**: Cloud-dependent; no offline deployment option; limited to API-based integration.
- **Relevance**: Similar to Bhashini — validates demand but doesn't solve the offline/
  feature-phone gap.

## 4. Research Gaps in Current Solutions

Across all three platforms, we identified four critical research gaps:

### Gap 1: Cloud Dependency
All existing solutions require internet connectivity. In rural India, where government
welfare schemes are most needed, internet connectivity is unreliable or unavailable.
Our offline-first approach with local ChromaDB + faster-whisper addresses this gap.

### Gap 2: Feature-Phone Exclusion
WhatsApp-based solutions (Jugalbandi) and smartphone apps (Bhashini/UMANG) exclude
citizens using feature phones. India has ~300 million feature phone users, many in
rural areas. Our IVR-compatible pipeline design (ASR → response → TTS) can serve
this population.

### Gap 3: LLM Hallucination on Government Policies
Jugalbandi uses LLMs for intent understanding and response generation, which can
produce incorrect or fabricated information about government schemes. Our grounded
RAG approach (ChromaDB retrieval + template-based response) ensures zero hallucination
by only returning information from verified government documents.

### Gap 4: High Operational Cost
Cloud-based solutions incur per-request costs that make free public service
unsustainable at scale. Our local computation model has zero marginal cost after
initial setup.

## 5. Problem Formulation — NLU/RAG Pipeline

Based on the literature review, the core NLU/RAG engineering problem for GovConnect is:

**Design a grounded, low-latency Retrieval-Augmented Generation system** that:
1. Uses ChromaDB for local vector storage (no cloud dependency)
2. Employs BAAI/bge-m3 for multilingual embedding (replacing gated IndicBERT)
3. Achieves >80% intent classification accuracy across 5 query types in 3 languages
4. Retrieves relevant scheme information with >70% precision@4
5. Generates zero-hallucination responses from verified document chunks
6. Operates entirely offline on consumer hardware (RTX 4050, 16GB RAM)

**Current status**:
- Intent classification: Nearest-centroid over bge-m3 embeddings (prototype)
- RAG retrieval: ChromaDB with top-4 L2 distance (prototype)
- Response generation: Template-based from retrieved chunks (zero-hallucination)
- Accuracy metrics: Pending formal evaluation (Week 3)

## 6. Our Competitive Position

| Capability | Jugalbandi | Bhashini | GovConnect |
|:--|:--|:--|:--|
| Offline operation | No | No | **Yes** |
| Feature-phone support | No (WhatsApp) | No (app) | **Yes (IVR)** |
| Hallucination risk | High (LLM) | Medium | **None (RAG + templates)** |
| Cost per query | Cloud compute | API fees | **Zero (local)** |
| Privacy | Data leaves device | Data leaves device | **Data stays local** |
| Scalability | High (cloud) | High (cloud) | Medium (local compute) |
| Accuracy | High | High | Medium (improving) |

Our positioning is complementary to existing solutions: we serve the population they
cannot reach (offline, feature-phone, privacy-sensitive), while they serve the
connected, smartphone-owning population.
