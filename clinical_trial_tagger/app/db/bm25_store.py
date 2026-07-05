import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    """In-memory BM25 index over ingested chunks. Rebuilt from Weaviate on app startup."""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        """Replace the index. chunks: [{"content": str, "category": str, "nct_id": str, ...}]"""
        self._corpus = chunks
        self._rebuild()

    def add_many(self, chunks: list[dict]) -> None:
        self._corpus.extend(chunks)
        self._rebuild()

    def _rebuild(self) -> None:
        if not self._corpus:
            self._bm25 = None
            return
        tokenized = [_tokenize(c["content"]) for c in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def query(self, text: str, top_k: int = 10) -> list[dict]:
        if self._bm25 is None or not self._corpus:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{**self._corpus[i], "score": float(scores[i])} for i in ranked_indices]

    def size(self) -> int:
        return len(self._corpus)


_bm25_store: BM25Store | None = None


def get_bm25_store() -> BM25Store:
    global _bm25_store
    if _bm25_store is None:
        _bm25_store = BM25Store()
    return _bm25_store
