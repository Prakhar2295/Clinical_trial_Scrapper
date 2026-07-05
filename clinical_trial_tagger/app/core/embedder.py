import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class Embedder:
    """Wraps a SentenceTransformer model for dense embeddings."""

    def __init__(self, model_name: str | None = None):
        self.model = SentenceTransformer(
            model_name or settings.embedding_model,
            trust_remote_code=True,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            task="retrieval.passage",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
