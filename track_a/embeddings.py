"""A2 - Embedding backends: IndicBERT (ai4bharat/indic-bert) with MiniLM fallback.

The plan requires IndicBERT. IndicBERT is a masked LM, so its CLS/mean-pooled
hidden states act as a sentence encoder for the few-shot classifier and for
ChromaDB indexing. If loading/embedding IndicBERT fails (e.g., download issue on
transformers 5.x), we automatically fall back to a MiniLM sentence encoder.
"""
from __future__ import annotations

import numpy as np

from shared.config import load_settings
from shared.logger import get_logger

log = get_logger(__name__)


class EmbeddingBackend:
    def __init__(self, backend: str = "auto", batch_size: int | None = None):
        cfg = load_settings()["track_a"]["embeddings"]
        self.backend_name = backend
        self.batch_size = batch_size or cfg["batch_size"]
        self._model = None
        self._loaded: str | None = None
        self._load()

    def _load(self) -> None:
        cfg = load_settings()["track_a"]["embeddings"]
        candidates: list[str] = []
        if self.backend_name == "indicbert":
            candidates = ["indicbert"]
        elif self.backend_name == "minilm":
            candidates = ["minilm"]
        else:  # auto
            candidates = ["indicbert", "minilm"]

        for cand in candidates:
            try:
                if cand == "indicbert":
                    self._load_indicbert(cfg["indicbert_model"])
                else:
                    self._load_minilm(cfg["minilm_model"])
                self._loaded = cand
                log.info("embedding backend ready: %s", cand)
                return
            except Exception as exc:
                log.warning("embedding backend %s failed (%s); trying next", cand, exc)

        raise RuntimeError("no embedding backend available")

    def _load_indicbert(self, model_name: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._use_torch = True
        self._model_name = model_name

    def _load_minilm(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._tokenizer = None
        self._use_torch = False
        self._model_name = model_name

    @property
    def backend(self) -> str | None:
        return self._loaded

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._use_torch:
            return self._embed_torch(texts)
        return np.asarray(self._model.encode(texts, batch_size=self.batch_size), dtype=np.float32)

    def _embed_torch(self, texts: list[str]) -> np.ndarray:
        import torch

        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tokenizer(batch, padding=True, truncation=True, max_length=128,
                                  return_tensors="pt")
            with torch.no_grad():
                out = self._model(**enc)
            vec = out.last_hidden_state.mean(dim=1).numpy()
            all_vecs.append(vec)
        return np.vstack(all_vecs).astype(np.float32)

    @property
    def dim(self) -> int:
        if self._use_torch:
            return int(self._model.config.hidden_size)
        return int(self._model.get_sentence_embedding_dimension())
