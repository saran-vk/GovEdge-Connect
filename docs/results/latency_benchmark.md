# Per-Stage and End-to-End Latency Benchmark Report

- **Total Benchmark Queries:** 24

### Latency Breakdown by Execution Path

| Execution Path | Queries | Avg Latency | p50 (Median) | p95 Latency | SLA Target | SLA Compliance |
|---|---|---|---|---|---|---|
| `cached` | 20 | 0.8 ms | 1.1 ms | 1.6 ms | < 2000 ms | **Pass** |
| `cloud_assisted` | 1 | 8.8 ms | 8.8 ms | 8.8 ms | < 3000 ms | **Pass** |
| `offline_extractive` | 3 | 0.0 ms | 0.0 ms | 0.0 ms | < 1500 ms | **Pass** |

### Per-Stage Average Latency Breakdown

| Path | ASR (ms) | NMT (ms) | NLU (ms) | RAG (ms) | LLM/Cache (ms) | TTS (ms) | Total (ms) |
|---|---|---|---|---|---|---|---|
| `cached` | 0.0 | 0.0 | 1.91 | 0.0 | 0.0 | 0.0 | 0.83 |
| `cloud_assisted` | 0.0 | 0.0 | 0.0 | 8.78 | 150.01 | 0.0 | 8.82 |
| `offline_extractive` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.02 |
