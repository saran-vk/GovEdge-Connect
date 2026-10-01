"""
Base NMT Adapter interface contract.
NMT: (text, src_lang, tgt_lang) -> {text, ms}
"""

from abc import ABC, abstractmethod
from typing import TypedDict


class NMTResult(TypedDict):
    text: str
    ms: float


class BaseNMTAdapter(ABC):
    @abstractmethod
    async def translate(self, text: str, src_lang: str, tgt_lang: str) -> NMTResult:
        """Translate text from src_lang to tgt_lang."""
        pass
