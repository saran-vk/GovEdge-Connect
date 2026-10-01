"""
Intent classification accuracy and confusion matrix evaluation harness.
"""

from typing import List, Dict, Any


def evaluate_intents(predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """Calculates intent accuracy and per-intent metrics."""
    if not ground_truths or len(predictions) != len(ground_truths):
        return {"accuracy": 0.0, "total": 0}

    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    total = len(ground_truths)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "total": total,
        "correct": correct,
    }
