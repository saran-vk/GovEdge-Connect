"""
Rule-based NLU adapter for mock/offline testing.
"""

import time
from services.nlu.base import BaseNLUAdapter, NLUResult
from gateway.schemas import Intent, IntentResult, Slots


class RulesNLUAdapter(BaseNLUAdapter):
    """Keyword/rule-based NLU for mock mode and baseline testing."""

    async def extract(self, text: str, lang: str = "en") -> NLUResult:
        t0 = time.perf_counter()
        lower = text.lower()

        # Slot detection: scheme
        scheme_name = None
        if "kisan" in lower or "pm-kisan" in lower or "pmkisan" in lower:
            scheme_name = "pm_kisan"
        elif "mgnrega" in lower or "nrega" in lower or "100 day" in lower:
            scheme_name = "mgnrega"
        elif "ayushman" in lower or "pmjay" in lower or "bharat" in lower or "card" in lower:
            scheme_name = "ayushman_bharat"

        # Intent detection
        if any(w in lower for w in ["eligible", "eligibility", "who can", "criteria"]):
            intent_val = Intent.CHECK_ELIGIBILITY
            conf = 0.92
        elif any(w in lower for w in ["document", "docs", "paper", "aadhaar", "pan"]):
            intent_val = Intent.DOCUMENTS_NEEDED
            conf = 0.90
        elif any(w in lower for w in ["apply", "process", "register", "how to"]):
            intent_val = Intent.APPLY_PROCESS
            conf = 0.88
        elif any(w in lower for w in ["amount", "benefit", "money", "rupees", "how much", "installment"]):
            intent_val = Intent.BENEFIT_AMOUNT
            conf = 0.91
        else:
            intent_val = Intent.OUT_OF_SCOPE
            conf = 0.50

        ms = (time.perf_counter() - t0) * 1000
        return {
            "intent": IntentResult(name=intent_val, confidence=conf),
            "slots": Slots(scheme_name=scheme_name),
            "ms": round(ms, 2),
        }
