"""
Intent classification accuracy, precision/recall, and confusion matrix evaluation harness.
Evaluates on data/intents/intent_dataset.json and verifies against the >=85% SLA target.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
import numpy as np

DATA_PATH = Path(__file__).parent.parent / "data" / "intents" / "intent_dataset.json"
RESULTS_DIR = Path(__file__).parent.parent / "docs" / "results"


def evaluate_intents(predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """Calculates intent accuracy, per-class metrics, and confusion matrix."""
    if not ground_truths or len(predictions) != len(ground_truths):
        return {"accuracy": 0.0, "total": 0}

    total = len(ground_truths)
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / total if total > 0 else 0.0

    classes = sorted(list(set(ground_truths + predictions)))
    confusion: Dict[str, Dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}
    for p, g in zip(predictions, ground_truths):
        confusion[g][p] += 1

    per_class = {}
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[other][c] for other in classes if other != c)
        fn = sum(confusion[c][other] for other in classes if other != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[c].values()),
        }

    return {
        "accuracy": round(accuracy, 4),
        "total": total,
        "correct": correct,
        "target_met": accuracy >= 0.85,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def run_intent_benchmark(save_results: bool = True) -> Dict[str, Any]:
    """Runs cross-validated intent benchmark using the trained pipeline and logs metrics."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [x["query"] for x in data]
    labels = [x["intent"] for x in data]

    union = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=1)),
        ("char", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=1)),
    ])
    pipeline = Pipeline([
        ("features", union),
        ("clf", LogisticRegression(C=10.0, max_iter=300)),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = [""] * len(labels)

    for train_idx, test_idx in skf.split(texts, labels):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]

        pipeline.fit(train_texts, train_labels)
        preds = pipeline.predict(test_texts)
        for idx, p in zip(test_idx, preds):
            all_preds[idx] = p

    results = evaluate_intents(all_preds, labels)

    if save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_file = RESULTS_DIR / "intent_benchmark.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        md_file = RESULTS_DIR / "intent_benchmark_report.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Intent Classification Benchmark Report\n\n")
            f.write(f"- **Total Samples:** {results['total']}\n")
            f.write(f"- **Overall Accuracy (5-Fold CV):** {results['accuracy'] * 100:.2f}%\n")
            f.write(f"- **Target Met ($\ge 85\%$):** {results['target_met']}\n\n")
            f.write("### Per-Intent Performance\n\n")
            f.write("| Intent | Precision | Recall | F1-Score | Support |\n")
            f.write("|---|---|---|---|---|\n")
            for c, m in results["per_class"].items():
                f.write(f"| `{c}` | {m['precision']*100:.1f}% | {m['recall']*100:.1f}% | {m['f1']*100:.1f}% | {m['support']} |\n")

        print(f"Saved Intent Benchmark to {json_file} and {md_file}")

    return results


if __name__ == "__main__":
    res = run_intent_benchmark(save_results=True)
    print(f"5-Fold CV Intent Accuracy: {res['accuracy'] * 100:.2f}% | Target Met: {res['target_met']}")
