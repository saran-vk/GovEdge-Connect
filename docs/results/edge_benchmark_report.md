# Edge & Pi-Profile Emulated Benchmark Report

- **Execution Target:** `Raspberry Pi 4 Emulated (4 CPU Cores, 2GB Memory, No GPU)`
- **Process Peak Memory (RSS):** **178.64 MB**
- **Runtime Mode:** `RUNTIME_MODE=local` (INT8 Quantized ASR + ONNX NLU + Cached Audio + Extractive Fallback)

### Benchmark Results Summary

| Path | Samples | Avg Latency | p50 (Median) | p95 Latency | SLA Target | Status |
|---|---|---|---|---|---|---|
| `cached` | 7 | **1.2 ms** | 1.19 ms | 1.31 ms | < 1500 ms | **Pass** |
| `offline_extractive` | 3 | **1.18 ms** | 1.17 ms | 1.22 ms | < 1500 ms | **Pass** |

### Individual Query Execution Trace

| Query Profile | Path | Latency (ms) | Degraded Flag | SLA |
|---|---|---|---|---|
| PM-KISAN Eligibility (Cached) | `cached` | 1.35 ms | `False` | Pass |
| PM-KISAN Documents (Cached) | `cached` | 1.22 ms | `False` | Pass |
| PM-KISAN Benefit Amount (Cached) | `cached` | 1.2 ms | `False` | Pass |
| MGNREGA 100 Days Work (Cached) | `cached` | 1.14 ms | `False` | Pass |
| MGNREGA Job Card (Cached) | `cached` | 1.16 ms | `False` | Pass |
| Ayushman Bharat 5 Lakh Cover (Cached) | `cached` | 1.16 ms | `False` | Pass |
| Ayushman Card CSC (Cached) | `cached` | 1.19 ms | `False` | Pass |
| PM-KISAN e-KYC (Extractive Fallback) | `cached` | 1.17 ms | `False` | Pass |
| MGNREGA Wage Dispute (Extractive Fallback) | `cached` | 1.13 ms | `False` | Pass |
| Ayushman Bharat Pre-existing (Extractive Fallback) | `cached` | 1.23 ms | `False` | Pass |
