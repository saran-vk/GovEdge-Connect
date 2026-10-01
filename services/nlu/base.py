"""
Base NLU Adapter interface contract.
Extracts intent and scheme/state slots from text.
"""

from abc import ABC, abstractmethod
from typing import TypedDict
from gateway.schemas import IntentResult, Slots


class NLUResult(TypedDict):
    intent: IntentResult
    slots: Slots
    ms: float


class BaseNLUAdapter(ABC):
    @abstractmethod
    async def extract(self, text: str, lang: str = "en") -> NLUResult:
        """Extract intent and slots from text."""
        pass
