# Scope Lock & Language Tiering Matrix

**Project:** GovConnect Edge  
**Review Target:** Prototype Milestone  
**Basis:** Master Prototype Plan (Single Source of Truth)

---

## 1. Locked Language Tiers

| Tier | Languages | Status | Commitment |
|---|---|---|---|
| **Tier 1** | **Tamil, Hindi** | Locked & Promised | Full pipeline: Dialect ASR, NMT/direct NLU, RAG, Cached TTS |
| **Tier 2** | **Telugu, Malayalam** | Stretch Candidates | Promoted only if passing all gates at Week 3 |

### Language Inclusion Gates (End of Week 3)
A Tier 2 language is promoted only if it passes all four gates:
1. **Data gate:** Sufficient IndicVoices audio available for validation/fine-tuning.
2. **ASR gate:** Baseline WER measured and acceptable within compute budget.
3. **TTS gate:** Confirmed working TTS voice (Bulbul or IndicTTS).
4. **Eval gate:** $\ge 50$ evaluation queries verified by a fluent speaker.

*Failure action:* If a gate fails, the language remains a text-only fallback or is cleanly excluded.

---

## 2. In-Scope Promises
- **Welfare Schemes (3):** PM-KISAN, MGNREGA, Ayushman Bharat.
- **Demo Surface:** Browser-based voice-to-voice interface showing transcript, intent, citations, execution path badge, and per-stage timings.
- **Execution Modes:** `RUNTIME_MODE=mock | hybrid | local`.
- **Metrics Evaluated:**
  - Dialect ASR WER ($\le 30\%$ on Tier 1 post-LoRA vs. baseline)
  - Intent Accuracy ($\ge 85\%$ on held-out set)
  - Hallucination Rate ($< 5\%$ on $\ge 50$ manually graded real queries)
  - Latency ($< 3.0\text{s}$ hybrid, $< 2.0\text{s}$ cached under Pi profile)
- **Edge Deployment:** INT8 quantization for ASR and NLU, benchmarked under Docker "Pi-profile" CPU constraints.

---

## 3. Explicitly Out of Scope
- Physical Raspberry Pi 4 hardware benchmarking (labeled honestly as emulated Pi-profile).
- Live telephony / IVR (Twilio/Asterisk stubs only).
- Additional languages beyond Tamil, Hindi, Telugu, Malayalam.
- Fine-tuning IndicTrans2 translation models.
- Running generative LLMs on-device.
