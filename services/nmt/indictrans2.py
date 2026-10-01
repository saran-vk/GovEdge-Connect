"""
IndicTrans2 NMT adapter placeholder stub.
"""

from services.nmt.base import BaseNMTAdapter, NMTResult


class IndicTrans2NMTAdapter(BaseNMTAdapter):
    """Production IndicTrans2 NMT adapter."""

    def __init__(self, model_name_or_path: str = "ai4bharat/indictrans2-indic-en-1B"):
        self.model_name_or_path = model_name_or_path
        self.model = None

    async def translate(self, text: str, src_lang: str, tgt_lang: str) -> NMTResult:
        raise NotImplementedError("IndicTrans2 adapter requires model loading in hybrid mode.")
