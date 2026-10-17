"""
Domain feature extractors for multilingual NLU classification.
Extracts domain scheme indicators and structural patterns for welfare intent classification.
"""

from typing import List
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

SCHEME_TERMS = [
    "kisan", "pm-kisan", "pmkisan", "farmer", "farmland", "cultivable", "unskilled", "panchayat", "gram", "beneficiary",
    "golden card", "health cover", "hospital", "medical cover", "wage", "allowance", "installment",
    "pmay", "pmay-g", "awas", "awaas", "housing", "pucca house", "kutcha",
    "கிசான்", "விவசாயி", "100 நாள்", "வேலை அட்டை", "ஆயுஷ்மான்", "பஞ்சாயத்து", "காப்பீடு", "ஆவாஸ்", "வீடு",
    "किसान", "मनरेगा", "जॉब कार्ड", "आयुष्मान", "मजदूरी", "किस्त", "पंचायत", "गोल्डन कार्ड", "आवास", "मकान",
    "రైతు", "ఉపాధి", "జాబ్ కార్డు", "ఆయుష్మాన్", "కిసాన్", "కూలీ", "ఆవాస్", "ఇల్లు",
    "കിസാൻ", "തൊഴിലുറപ്പ്", "ജോബ് കാർഡ്", "ആയുഷ്മാൻ", "വേതനം", "ആവാസ്", "വീട്",
]


class DomainKeywordFeature(BaseEstimator, TransformerMixin):
    """Extracts scheme keyword counts, absence indicators, and token length."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]):
        features = []
        for text in X:
            t = text.lower()
            scheme_hits = sum(1.0 for term in SCHEME_TERMS if term in t)
            q_len = len(t.split())
            features.append([scheme_hits, 1.0 if scheme_hits == 0 else 0.0, float(q_len)])
        return np.array(features)
