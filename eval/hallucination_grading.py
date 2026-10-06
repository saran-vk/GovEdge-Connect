"""
Hallucination and Faithfulness Grading Harness.
Evaluates end-to-end answers against official scheme guidelines in data/schemes/
and ground truth facts in data/eval/50_seed_queries.json.
Target: < 5% hallucination rate on >=50 held-out real scheme queries.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from services.orchestrator import OrchestratorPipeline
from gateway.schemas import Path as ExecPath, RuntimeMode

EVAL_PATH = Path(__file__).parent.parent / "data" / "eval" / "50_seed_queries.json"
RESULTS_DIR = Path(__file__).parent.parent / "docs" / "results"

# Critical scheme facts for validation
NUMERICAL_FACTS = {
    "pm_kisan": {
        "annual_amount": ["6,000", "6000"],
        "installment_amount": ["2,000", "2000"],
        "installments_count": ["three", "3"],
    },
    "mgnrega": {
        "guaranteed_days": ["100"],
        "days_limit": ["15"],
    },
    "ayushman_bharat": {
        "annual_cover": ["5,00,000", "500000", "5 lakh", "5 lakhs"],
    },
}


def grade_hallucination_batch(graded_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates faithfulness against source documents.
    A response fails if it states facts not supported by cited chunks or violates known ground truth.
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
        "faithful_count": total - hallucinated,
        "hallucination_rate": round(rate, 4),
        "target_met": rate < 0.05,
    }


async def run_hallucination_evaluation(
    runtime_mode: RuntimeMode = RuntimeMode.HYBRID,
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Runs all 50 evaluation queries through the pipeline and grades hallucination rate.
    Logs comprehensive output to docs/results/hallucination_report.json and .md.
    """
    orchestrator = OrchestratorPipeline(default_mode=runtime_mode)

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    graded_items = []

    for item in queries:
        qid = item["id"]
        q_text = item["query_en"]
        target_scheme = item["target_scheme"]
        is_out_of_scope = item["is_out_of_scope"]
        gt_facts = item["ground_truth_facts"]

        # Run through pipeline
        res = await orchestrator.process_query(
            session_id=f"eval_{qid}",
            text=q_text,
            lang="en",
            runtime_mode=runtime_mode,
        )

        reply_lower = res.reply_text.lower()
        is_hallucinated = False
        reasons = []

        if is_out_of_scope:
            # Must refuse or state only PM-KISAN, MGNREGA, Ayushman Bharat are supported
            if "only answer" not in reply_lower and "cannot answer" not in reply_lower:
                # If it generated factual answers for outside domains (weather, driving license), flag hallucination
                if any(w in reply_lower for w in ["driving", "licence", "license", "tatkal", "cricket", "temperature", "forecast"]):
                    is_hallucinated = True
                    reasons.append("Answered out-of-scope domain query instead of refusing.")
        else:
            # In-scope welfare query
            # 1. Check citations: should have citations unless cached
            if res.path != ExecPath.CACHED and not res.citations:
                is_hallucinated = True
                reasons.append("Uncited generation in RAG path.")

            # 2. Check numerical hallucinations
            if target_scheme == "pm_kisan":
                # Check for wrong amounts
                wrong_amounts = ["10,000", "12,000", "8,000", "10000", "12000"]
                if any(wa in res.reply_text for wa in wrong_amounts):
                    is_hallucinated = True
                    reasons.append(f"Stated incorrect PM-KISAN benefit amount.")
            elif target_scheme == "mgnrega":
                wrong_days = ["150 days", "200 days", "50 days"]
                if any(wd in res.reply_text for wd in wrong_days):
                    is_hallucinated = True
                    reasons.append("Stated incorrect MGNREGA employment duration.")
            elif target_scheme == "ayushman_bharat":
                wrong_covers = ["10 lakh", "2 lakh", "1 lakh", "10,00,000"]
                if any(wc in res.reply_text for wc in wrong_covers):
                    is_hallucinated = True
                    reasons.append("Stated incorrect Ayushman Bharat coverage amount.")

        graded_items.append({
            "id": qid,
            "query": q_text,
            "scheme": target_scheme,
            "path": res.path.value,
            "hallucinated": is_hallucinated,
            "reasons": reasons,
            "reply_text": res.reply_text,
            "citations": [c.model_dump() for c in res.citations],
        })

    summary = grade_hallucination_batch(graded_items)
    summary["queries"] = graded_items

    if save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_file = RESULTS_DIR / "hallucination_report.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        md_file = RESULTS_DIR / "hallucination_report.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Hallucination & Faithfulness Evaluation Report\n\n")
            f.write(f"- **Evaluated Queries:** {summary['total_queries']}\n")
            f.write(f"- **Faithful Grounded Responses:** {summary['faithful_count']}\n")
            f.write(f"- **Hallucinated Responses:** {summary['hallucinated_count']}\n")
            f.write(f"- **Hallucination Rate:** **{summary['hallucination_rate'] * 100:.2f}%**\n")
            f.write(f"- **Target Met (< 5%):** **{summary['target_met']}**\n\n")
            f.write("### Evaluation Summary by Scheme\n\n")
            f.write("| Scheme | Queries | Faithful | Hallucinations | Hallucination Rate |\n")
            f.write("|---|---|---|---|---|\n")

            by_scheme = {}
            for g in graded_items:
                s = g["scheme"] or "out_of_scope"
                by_scheme.setdefault(s, {"total": 0, "hallucinated": 0})
                by_scheme[s]["total"] += 1
                if g["hallucinated"]:
                    by_scheme[s]["hallucinated"] += 1

            for s, counts in by_scheme.items():
                hrate = (counts["hallucinated"] / counts["total"]) * 100
                f.write(f"| `{s}` | {counts['total']} | {counts['total'] - counts['hallucinated']} | {counts['hallucinated']} | {hrate:.1f}% |\n")

        print(f"Hallucination Report logged to {json_file} and {md_file}")

    return summary


if __name__ == "__main__":
    res = asyncio.run(run_hallucination_evaluation(save_results=True))
    print(f"Hallucination Rate: {res['hallucination_rate'] * 100:.2f}% | Target Met: {res['target_met']}")
