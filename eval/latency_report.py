"""
End-to-end and per-stage latency benchmarking harness.
Evaluates queries across Cached, Cloud-assisted, and Offline-extractive paths.
Computes p50, p95, average, and SLA adherence, logging results to docs/results/.
"""

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from gateway.schemas import RuntimeMode, Path as ExecPath
from services.orchestrator import OrchestratorPipeline

RESULTS_DIR = Path(__file__).parent.parent / "docs" / "results"

BENCHMARK_QUERIES = [
    # Cached Queries (High confidence common intents for all 3 schemes)
    {"text": "Who is eligible for PM KISAN scheme?", "lang": "en", "mode": RuntimeMode.MOCK, "expected_path": "cached"},
    {"text": "What documents are needed for PM KISAN?", "lang": "en", "mode": RuntimeMode.MOCK, "expected_path": "cached"},
    {"text": "How do I apply for PM KISAN portal?", "lang": "en", "mode": RuntimeMode.MOCK, "expected_path": "cached"},
    {"text": "What is the annual benefit amount of PM KISAN?", "lang": "en", "mode": RuntimeMode.MOCK, "expected_path": "cached"},
    {"text": "Who is eligible for 100 days work under MGNREGA?", "lang": "en", "mode": RuntimeMode.MOCK, "expected_path": "cached"},
    {"text": "What documents are required to get an MGNREGA Job Card?", "lang": "en", "mode": RuntimeMode.MOCK, "expected_path": "cached"},
    {"text": "How to apply for job card in Gram Panchayat?", "lang": "en", "mode": RuntimeMode.MOCK, "expected_path": "cached"},
    {"text": "How many days of wage employment in MGNREGA?", "lang": "en", "mode": RuntimeMode.MOCK, "expected_path": "cached"},
    {"text": "Who is eligible for Ayushman Bharat PMJAY?", "lang": "en", "mode": RuntimeMode.MOCK, "expected_path": "cached"},
    {"text": "What documents are needed for Ayushman Golden Card?", "lang": "en", "mode": RuntimeMode.MOCK, "expected_path": "cached"},

    # Novel / Cloud-Assisted Queries (Cache misses triggering RAG retrieval & responder)
    {"text": "Can income tax paying farmers receive PM KISAN installments?", "lang": "en", "mode": RuntimeMode.HYBRID, "expected_path": "cloud_assisted"},
    {"text": "Are senior citizens aged 70 and above eligible for Ayushman Bharat?", "lang": "en", "mode": RuntimeMode.HYBRID, "expected_path": "cloud_assisted"},
    {"text": "Is unemployment allowance paid if MGNREGA work is delayed beyond 15 days?", "lang": "en", "mode": RuntimeMode.HYBRID, "expected_path": "cloud_assisted"},
    {"text": "Can tenant farmers apply without landholding documents?", "lang": "en", "mode": RuntimeMode.HYBRID, "expected_path": "cloud_assisted"},
    {"text": "Does Ayushman Bharat cover pre-existing diseases from day one?", "lang": "en", "mode": RuntimeMode.HYBRID, "expected_path": "cloud_assisted"},
    {"text": "Are equal wages paid to women and men under MGNREGA?", "lang": "en", "mode": RuntimeMode.HYBRID, "expected_path": "cloud_assisted"},
    {"text": "What is the distance limit for MGNREGA work from the village?", "lang": "en", "mode": RuntimeMode.HYBRID, "expected_path": "cloud_assisted"},
    {"text": "Does Ayushman Bharat cover medicine costs after hospital discharge?", "lang": "en", "mode": RuntimeMode.HYBRID, "expected_path": "cloud_assisted"},

    # Offline / Extractive Fallback Queries (Local mode or out of scope refusals)
    {"text": "What are the rules for PM KISAN beneficiary registration?", "lang": "en", "mode": RuntimeMode.LOCAL, "expected_path": "offline_extractive"},
    {"text": "How does Gram Panchayat issue job card within 15 days?", "lang": "en", "mode": RuntimeMode.LOCAL, "expected_path": "offline_extractive"},
    {"text": "What is the procedure for secondary hospitalization in PM-JAY?", "lang": "en", "mode": RuntimeMode.LOCAL, "expected_path": "offline_extractive"},
    {"text": "What is the weather forecast for tomorrow?", "lang": "en", "mode": RuntimeMode.MOCK, "expected_path": "offline_extractive"},
    {"text": "How to book tatkal train ticket on IRCTC portal?", "lang": "en", "mode": RuntimeMode.MOCK, "expected_path": "offline_extractive"},
    {"text": "Who won the cricket match yesterday between India and Australia?", "lang": "en", "mode": RuntimeMode.MOCK, "expected_path": "offline_extractive"},
]


def generate_latency_report(query_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarizes p50, p95, avg, and stage latency breakdown by execution path."""
    paths_data = {"cached": [], "cloud_assisted": [], "offline_extractive": []}

    for run in query_runs:
        p = run["path"]
        if p in paths_data:
            paths_data[p].append(run)

    report: Dict[str, Any] = {"summary": {}, "runs": query_runs}

    # SLA thresholds (ms)
    sla_limits = {
        "cached": 2000.0,
        "cloud_assisted": 3000.0,
        "offline_extractive": 1500.0,
    }

    for path_name, runs in paths_data.items():
        if not runs:
            report["summary"][path_name] = {"count": 0}
            continue

        totals = [r["timings_ms"]["total"] for r in runs]
        p50 = float(np.percentile(totals, 50))
        p95 = float(np.percentile(totals, 95))
        avg = float(np.mean(totals))
        limit = sla_limits[path_name]

        # Stage breakdown averages
        stages = ["asr", "nmt", "nlu", "rag", "llm_or_cache", "tts"]
        stage_avgs = {}
        for s in stages:
            s_vals = [r["timings_ms"].get(s, 0.0) or 0.0 for r in runs]
            stage_avgs[s] = round(float(np.mean(s_vals)), 2)

        report["summary"][path_name] = {
            "count": len(runs),
            "avg_ms": round(avg, 2),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "min_ms": round(min(totals), 2),
            "max_ms": round(max(totals), 2),
            "sla_target_ms": limit,
            "within_sla": p95 <= limit,
            "stages_avg_ms": stage_avgs,
        }

    return report


async def run_latency_benchmark(save_results: bool = True) -> Dict[str, Any]:
    """Executes the benchmark query suite and generates comprehensive reports."""
    orchestrator = OrchestratorPipeline()
    query_runs = []

    for i, item in enumerate(BENCHMARK_QUERIES):
        session_id = f"latency_bench_{i:03d}"
        res = await orchestrator.process_query(
            session_id=session_id,
            text=item["text"],
            lang=item.get("lang", "en"),
            runtime_mode=item["mode"],
        )
        query_runs.append({
            "query": item["text"],
            "path": res.path.value,
            "timings_ms": res.timings_ms.model_dump(),
            "within_sla": res.within_sla,
        })

    report = generate_latency_report(query_runs)

    if save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_file = RESULTS_DIR / "latency_benchmark.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        md_file = RESULTS_DIR / "latency_benchmark.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Per-Stage and End-to-End Latency Benchmark Report\n\n")
            f.write(f"- **Total Benchmark Queries:** {len(query_runs)}\n\n")
            f.write("### Latency Breakdown by Execution Path\n\n")
            f.write("| Execution Path | Queries | Avg Latency | p50 (Median) | p95 Latency | SLA Target | SLA Compliance |\n")
            f.write("|---|---|---|---|---|---|---|\n")

            for path_name, stats in report["summary"].items():
                if stats["count"] == 0:
                    continue
                sla_badge = "Pass" if stats["within_sla"] else "Warning"
                f.write(
                    f"| `{path_name}` | {stats['count']} | {stats['avg_ms']:.1f} ms | "
                    f"{stats['p50_ms']:.1f} ms | {stats['p95_ms']:.1f} ms | "
                    f"< {stats['sla_target_ms']:.0f} ms | **{sla_badge}** |\n"
                )

            f.write("\n### Per-Stage Average Latency Breakdown\n\n")
            f.write("| Path | ASR (ms) | NMT (ms) | NLU (ms) | RAG (ms) | LLM/Cache (ms) | TTS (ms) | Total (ms) |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for path_name, stats in report["summary"].items():
                if stats["count"] == 0:
                    continue
                st = stats["stages_avg_ms"]
                f.write(
                    f"| `{path_name}` | {st['asr']} | {st['nmt']} | {st['nlu']} | "
                    f"{st['rag']} | {st['llm_or_cache']} | {st['tts']} | {stats['avg_ms']} |\n"
                )

        print(f"Latency report logged to {json_file} and {md_file}")

    return report


if __name__ == "__main__":
    res = asyncio.run(run_latency_benchmark(save_results=True))
    for p, stats in res["summary"].items():
        if stats["count"] > 0:
            print(f"[{p}] count={stats['count']}, avg={stats['avg_ms']}ms, p95={stats['p95_ms']}ms, SLA={stats['within_sla']}")
