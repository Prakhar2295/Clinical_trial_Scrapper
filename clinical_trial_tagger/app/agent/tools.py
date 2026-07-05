from langchain_core.tools import tool

from app.core.embedder import Embedder
from app.core.reranker import Reranker
from app.db.bm25_store import get_bm25_store
from app.db.weaviate_client import get_weaviate_store

_embedder = Embedder()
_reranker = Reranker()


@tool
def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid dense+BM25 search over previously ingested clinical trial chunks.

    Returns the best-matching chunks, each with its content, category label, and filename.
    """
    bm25 = get_bm25_store()
    bm25_results = bm25.query(query, top_k=top_k)

    dense_results: list[dict] = []
    try:
        store = get_weaviate_store()
        if store.is_ready():
            vector = _embedder.embed_one(query).tolist()
            dense_results = store.query_near_vector(vector, limit=top_k)
    except Exception:
        dense_results = []

    combined: dict[str, dict] = {r["uuid"]: r for r in dense_results if r.get("uuid")}
    for r in bm25_results:
        combined.setdefault(r.get("uuid") or r["content"], r)

    candidates = list(combined.values())
    if not candidates:
        return []

    ranked = _reranker.rerank(query, [c["content"] for c in candidates])
    return [
        {
            "content": candidates[i]["content"],
            "category": candidates[i].get("category"),
            "filename": candidates[i].get("filename"),
            "score": score,
        }
        for i, score in ranked[:top_k]
    ]


AGENT_TOOLS = [hybrid_search]
