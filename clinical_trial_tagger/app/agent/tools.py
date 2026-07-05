from langchain_core.tools import tool

from app.core.embedder import Embedder
from app.db.weaviate_client import get_weaviate_store

_embedder = Embedder()


@tool
def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid search over previously ingested clinical trial chunks.

    Returns the best-matching chunks, each with its content, category label, and filename.
    """
    store = get_weaviate_store()
    if not store.is_ready():
        return []

    try:
        vector = _embedder.embed_one(query).tolist()
        return store.query_hybrid(
            query_text=query,
            vector=vector,
            chunk_position="head",
            limit=top_k,
        )
    except Exception as e:
        return [
            {
                "error": f"hybrid_search tool failed: {e}",
                "content": "",
                "category": "",
                "score": 0.0,
                "chunk_position": "head",
            }
        ]


AGENT_TOOLS = [hybrid_search]
