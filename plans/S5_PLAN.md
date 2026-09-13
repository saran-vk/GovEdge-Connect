# WeeK 2:

- Literature Review: Conducted a literature review on speech processing and translation frameworks, focusing on AI4Bharat IndicVoices, IndicASR, Bhashini, and Sarvam AI speech models. Identified major gaps in existing speech architectures: general-purpose translation layers lack domain-specific dialogue context, and current models struggle with rural colloquial accents and phonetic deviations.
    
- Problem Formulation: Formulated the core speech engineering problem statement. Focused on addressing high Word Error Rates (WER) in ASR models when handling regional Indian dialects, and optimizing the multi-stage voice processing pipeline (ASR to TTS) to achieve low-latency audio response synthesis over telephony and IVR networks.

- Literature Review: Conducted a literature review on conversational e-Governance architectures, analyzing systems like Jugalbandi, Bhashini, and Sarvam AI. Identified critical research gaps in current solutions: heavy reliance on commercial cloud platforms, high operation costs, WhatsApp dependency excluding citizens using feature phones, and frequent hallucinations in standard LLMs when handling complex government welfare policies.

- Problem Formulation: Formulated the core engineering problem for the NLU and RAG pipeline. Focused on designing a grounded, low-latency Retrieval-Augmented Generation system using ChromaDB vector database and IndicBERT intent classification to deliver factual, zero-hallucination scheme details on low-bandwidth edge devices.

# WeeK 3:

- Detailed Literature Review & Comparative Analysis: Expanded the literature review on vector database models and Retrieval-Augmented Generation (RAG) frameworks suitable for low-resource Indian languages. Compared ChromaDB and FAISS for vector embedding storage, metadata filtering capabilities, and retrieval performance. Analyzed IndicBERT and EmoBERTa architectures for extracting intent from noisy, translated speech transcripts.

- System Design & Problem Formulation: Formulated the technical system design for the grounded RAG engine. Structured the document ingestion pipeline to chunk government welfare scheme PDFs (such as PM-KISAN and MGNREGA) into semantic blocks with associated metadata tags to ensure high-precision retrieval and prevent LLM hallucinations.

- Speech Corpus Literature Review & Model Selection: Extended the literature review on acoustic speech recognition (ASR) fine-tuning strategies for localized Dravidian and Indic dialects. Analyzed fine-tuning methodologies for Whisper and Wav2Vec 2.0 architectures using low-resource dialect speech datasets. Investigated neural machine translation (NMT) via IndicTransv2 for routing multi-dialect queries smoothly.

- Speech Pipeline Formulation & Technical Design: Formulated the multi-stage speech processing architecture connecting ASR, translation, and TTS engines. Designed the data augmentation strategy to simulate background noise and phone call audio degradation to train robust acoustic models capable of achieving over 70% accent adaptability.

# WeeK 4:

- Theoretical & Mathematical Strategy Analysis: Analyzed the mathematical formulation of vector similarity retrieval and NLU intent classification. Computed cosine similarity metrics between query embedding vectors and stored document chunk embeddings in ChromaDB to maximize semantic relevance. Derived chunking math using a 512-character window with a 64-character sliding overlap to preserve contextual continuity across policy boundaries. Analyzed 8-bit model weight quantization (FP16 to INT8 mapping) to compress deep neural networks by 60%, ensuring model memory fit within the local 6GB GPU VRAM budget.

- Hardware & Pipeline Architecture Analysis: Designed the hardware deployment strategy optimized for local execution on the target workstation (AMD Ryzen 7 7735HS CPU, 16GB RAM, NVIDIA RTX 4050 GPU with PyTorch CUDA acceleration). Formulated containerized FastAPI microservices for handling API requests and running local ChromaDB vector retrieval efficiently.

- Theoretical & Mathematical Strategy Analysis: Formulated the mathematical evaluation framework for Speech Recognition (ASR) performance. Defined Word Error Rate (WER = (S + D + I) / N) and Character Error Rate (CER) equations to benchmark Whisper and Wav2Vec 2.0 acoustic models across regional dialects. Analyzed transformer self-attention mathematical mechanisms within IndicTransv2 for neural machine translation and calculated synthesis latency bounds for mms-tts audio generation.

- Hardware & Pipeline Architecture Analysis: Designed the multi-stage voice processing execution strategy optimized for local workstation hardware. Configured faster-whisper and TTS model pipelines to utilize PyTorch CUDA acceleration on the NVIDIA RTX 4050 GPU (6GB VRAM) and CPU multithreading (AMD Ryzen 7) to achieve real-time speech processing and sub-1.0s inference latency.
