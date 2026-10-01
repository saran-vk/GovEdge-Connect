"""
IndicBERTv2-SS fine-tuned NLU adapter stub.
"""

from services.nlu.base import BaseNLUAdapter, NLUResult


class IndicBERTNLUAdapter(BaseNLUAdapter):
    """Production fine-tuned IndicBERTv2-SS intent classifier & slot filler."""

    def __init__(self, model_dir_or_name: str = "models/indicbert-nlu"):
        self.model_dir_or_name = model_dir_or_name
        self.model = None

    async def extract(self, text: str, lang: str = "en") -> NLUResult:
        raise NotImplementedError("IndicBERTNLUAdapter requires trained weights loaded in hybrid/local mode.")
