"""A3 - NLU: few-shot intent classification + rule-based entity extraction.

Intent classification is a nearest-centroid classifier over embedding space:
  * exemplar queries per intent (from taxonomy.INTENT_ALIASES + seed examples)
  * centroid = mean embedding of exemplars
  * query -> nearest centroid by cosine similarity
This is intentionally a prototype: it proves the taxonomy pipeline without a
labelled fine-tuning set. A future Week-2 task swaps in a trained head.

Entity extraction is rule-based (regex/keywords) covering the six annotated
entity types, including basic transliterated Hindi/Tamil matches.
"""
from __future__ import annotations

import re

import numpy as np

from shared.logger import get_logger

from track_a.embeddings import EmbeddingBackend
from track_a.taxonomy import ENTITIES, ENTITY_PATTERNS, INTENTS, SCHEME_KEYWORDS

log = get_logger(__name__)

EXEMPLARS: dict[str, list[str]] = {
    "check_eligibility": [
        "am I eligible for PM Kisan",
        "who can apply for Ayushman Bharat",
        "do I qualify for the housing scheme",
        "income limit for PM Awas Yojana",
        "can a farmer above 60 get the benefit",
        "kya main pm kisan ke liye patra hun",
        "எனக்கு pm kisan தகுதி உண்டா",
    ],
    "scheme_benefits": [
        "how much money does PM Kisan give",
        "what are the benefits of Ayushman Bharat",
        "payout amount per year",
        "what assistance does the housing scheme provide",
        "kitna paisa milega",
        "இதில் என்ன பலன் கிடைக்கும்",
    ],
    "required_documents": [
        "which documents are required for PM Kisan",
        "what documents do I need to apply",
        "documents needed for Ayushman Bharat",
        "do I need Aadhaar and a land certificate",
        "kya documents chahiye",
        "என்ன ஆவணங்கள் தேவை",
    ],
    "application_status": [
        "how do I check my PM Kisan status",
        "track application status",
        "when will my application be approved",
        "where to check the status online",
        "mera aavedan kahan hai",
        "என் விண்ணப்ப நிலை என்ன",
    ],
    "general_inquiry": [
        "what is PM Kisan scheme",
        "tell me about Ayushman Bharat",
        "general information on rural housing",
        "how can I get more information",
        "is scheme ke baare me batao",
        "இந்த திட்டம் பற்றி சொல்லுங்கள்",
    ],
}


class IntentClassifier:
    def __init__(self, embedding_backend: EmbeddingBackend | None = None):
        self.embeddings = embedding_backend or EmbeddingBackend()
        self._centroids: dict[str, np.ndarray] | None = None

    def _build_centroids(self) -> None:
        centroids: dict[str, np.ndarray] = {}
        for intent, examples in EXEMPLARS.items():
            vecs = self.embeddings.embed(examples)
            centroids[intent] = vecs.mean(axis=0)
            centroids[intent] /= np.linalg.norm(centroids[intent])
        self._centroids = centroids

    def classify(self, query: str) -> tuple[str, dict[str, float]]:
        if self._centroids is None:
            self._build_centroids()
        qvec = self.embeddings.embed([query])[0]
        qnorm = qvec / (np.linalg.norm(qvec) + 1e-9)
        scores: dict[str, float] = {}
        for intent, centroid in self._centroids.items():
            scores[intent] = float(np.dot(qnorm, centroid))
        best = max(scores, key=scores.get)
        return best, scores

    def extract_entities(self, query: str) -> dict[str, list[str]]:
        entities: dict[str, list[str]] = {k: [] for k in ENTITIES}
        text = query.lower()

        for kind, pattern in ENTITY_PATTERNS.items():
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                val = next((g for g in m.groups() if g), m.group(0))
                if val and val not in entities[kind]:
                    entities[kind].append(val)

        for scheme, kws in SCHEME_KEYWORDS.items():
            if any(kw in text for kw in kws):
                if scheme not in entities["scheme_name"]:
                    entities["scheme_name"].append(scheme)

        return {k: v for k, v in entities.items() if v}

    def analyse(self, query: str) -> dict:
        intent, scores = self.classify(query)
        return {
            "intent": intent,
            "intent_scores": scores,
            "entities": self.extract_entities(query),
            "query": query,
        }
