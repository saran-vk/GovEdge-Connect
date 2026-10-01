"""
Per-stage latency tracking middleware.
Aggregates stage execution times and assesses SLA compliance (<3.0s hybrid, <1.5s cached).
"""

import time
from typing import Dict
from gateway.schemas import TimingsMs, Path


class LatencyTracker:
    """Tracks latency across pipeline stages."""

    SLA_THRESHOLDS_MS = {
        Path.CACHED: 2000.0,
        Path.CLOUD_ASSISTED: 3000.0,
        Path.OFFLINE_EXTRACTIVE: 1500.0,
    }

    def __init__(self):
        self.stage_timings: Dict[str, float] = {
            "asr": 0.0,
            "nmt": 0.0,
            "nlu": 0.0,
            "rag": 0.0,
            "llm_or_cache": 0.0,
            "tts": 0.0,
        }
        self.start_time = time.perf_counter()

    def record_stage(self, stage: str, duration_ms: float):
        """Record explicit elapsed milliseconds reported by adapter."""
        if stage in self.stage_timings:
            self.stage_timings[stage] = round(duration_ms, 2)

    def finalize(self, path: Path) -> Tuple_Timings:
        total_ms = (time.perf_counter() - self.start_time) * 1000
        timings = TimingsMs(
            asr=self.stage_timings.get("asr", 0.0),
            nmt=self.stage_timings.get("nmt", 0.0),
            nlu=self.stage_timings.get("nlu", 0.0),
            rag=self.stage_timings.get("rag", 0.0),
            llm_or_cache=self.stage_timings.get("llm_or_cache", 0.0),
            tts=self.stage_timings.get("tts", 0.0),
            total=round(total_ms, 2),
        )
        threshold = self.SLA_THRESHOLDS_MS.get(path, 3000.0)
        within_sla = total_ms <= threshold
        return timings, within_sla


from typing import Tuple
Tuple_Timings = Tuple[TimingsMs, bool]
