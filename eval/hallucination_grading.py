"""
Hallucination / Faithfulness manual and automated grading rubric harness.
Target: <5% hallucination rate on >=50 held-out real scheme queries.
"""

from typing import List, Dict, Any


def grade_hallucination_batch(graded_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates faithfulness against source documents.
    A response fails if it states facts not supported by a cited chunk.
    Out-of-scope refusals count as correct.
    """
    total = len(graded_results)
    if total == 0:
        return {"hallucination_rate": 0.0, "total": 0, "hallucinated_count": 0}

    hallucinated = sum(1 for item in graded_results if item.get("hallucinated", False))
    rate = hallucinated / total

    return {
        "total_queries": total,
        "hallucinated_count": hallucinated,
        "hallucination_rate": round(rate, 4),
        "target_met": rate < 0.05,
    }
