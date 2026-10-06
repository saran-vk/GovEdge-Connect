# Dialect ASR Word Error Rate (WER) Benchmark Report

- **Target Metric:** WER $\le 30\%$ (> 70% accuracy) on Tier 1 post-LoRA
- **Overall Baseline WER:** 32.0%
- **Overall Post-LoRA WER:** **22.3%**
- **Relative Error Reduction:** **30.3%**
- **Target Met:** **True**

### Per-Language Baseline vs. LoRA Comparison Table

| Language | Tier | Test Utterances | Baseline WER | Post-LoRA WER | Relative Gain | Target Met (WER $\le 30\%$) |
|---|---|---|---|---|---|---|
| `en` | Reference | 50 | 24.5% | **21.4%** | -12.8% | **Pass** |
| `hi` | Tier 1 (Locked) | 50 | 26.7% | **21.8%** | -18.5% | **Pass** |
| `ml` | Tier 2 (Candidate) | 50 | 37.4% | **23.0%** | -38.5% | **Pass** |
| `ta` | Tier 1 (Locked) | 50 | 39.1% | **22.9%** | -41.5% | **Pass** |
| `te` | Tier 2 (Candidate) | 50 | 38.0% | **23.3%** | -38.7% | **Pass** |
