from sentence_transformers import CrossEncoder

from app.core.config import settings


class Reranker:
    """Wraps a cross-encoder model to rerank (query, document) pairs."""

    def __init__(self, model_name: str | None = None):
        self.model = CrossEncoder(model_name or settings.reranker_model)

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """Returns (original_index, score) pairs sorted by descending relevance."""
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked_indices = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in ranked_indices]
