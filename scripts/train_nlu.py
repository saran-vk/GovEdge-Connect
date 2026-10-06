"""
Train multilingual Intent Classification Pipeline on data/intents/intent_dataset.json
and save model artifacts to models/indicbert_nlu.joblib.
"""

import json
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

DATA_PATH = Path(__file__).parent.parent / "data" / "intents" / "intent_dataset.json"
MODEL_PATH = Path(__file__).parent.parent / "models" / "indicbert_nlu.joblib"


def train_nlu():
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

    pipeline.fit(texts, labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    train_acc = pipeline.score(texts, labels)
    print(f"Trained NLU model on {len(texts)} samples with train accuracy: {train_acc * 100:.2f}%")
    print(f"Saved model to: {MODEL_PATH}")
    return pipeline


if __name__ == "__main__":
    train_nlu()
