from langchain_core.tools import tool
from weaviate.classes.query import MetadataQuery

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

    vector = _embedder.embed_one(query).tolist()
    result = store.collection.query.hybrid(
        query=query,
        vector=vector,
        limit=top_k,
        return_metadata=MetadataQuery(score=True),
    )

    return [
        {
            "content": obj.properties.get("content", ""),
            "category": obj.properties.get("category"),
            "filename": obj.properties.get("filename"),
            "score": obj.metadata.score,
        }
        for obj in result.objects
    ]


AGENT_TOOLS = [hybrid_search]
