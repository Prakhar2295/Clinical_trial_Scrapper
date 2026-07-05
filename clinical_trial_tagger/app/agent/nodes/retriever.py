from collections import defaultdict

from weaviate.classes.query import Filter, MetadataQuery, Rerank

from app.agent.state import AgentState
from app.core.config import settings
from app.core.embedder import Embedder
from app.db.weaviate_client import get_weaviate_store

POSITION_ORDER = ("head", "tail")

_embedder = Embedder()


def _group_by_position(chunks: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        groups[chunk.get("chunk_position", "head")].append(chunk)
    return groups


def _search_position_group(collection, chunk_position: str, group: list[dict]) -> list[dict]:
    chunk_text = "\n".join(c["content"] for c in group)[:2000]
    vector = _embedder.embed_one(chunk_text).tolist()

    result = collection.query.hybrid(
        query=chunk_text,
        vector=vector,
        limit=settings.top_k_retrieval,
        filters=Filter.by_property("chunk_position").equal(chunk_position),
        rerank=Rerank(prop="content", query=chunk_text),
        return_metadata=MetadataQuery(score=True),
    )

    return [
        {
            "content": obj.properties.get("content", ""),
            "category": obj.properties.get("category"),
            "score": obj.metadata.score,
            "filename": obj.properties.get("filename"),
            "chunk_position": obj.properties.get("chunk_position"),
            "chunk_index": obj.properties.get("chunk_index"),
        }
        for obj in result.objects
    ]


def retriever_node(state: AgentState) -> dict:
    chunks = state.get("chunks") or []
    if not chunks:
        return {"retrieved_chunks": []}

    try:
        store = get_weaviate_store()
        if not store.is_ready():
            return {"error": "retriever_node: Weaviate unavailable", "retrieved_chunks": []}
    except Exception as exc:
        return {"error": f"retriever_node: Weaviate unavailable ({exc})", "retrieved_chunks": []}

    try:
        groups = _group_by_position(chunks)

        all_results: list[dict] = []
        for chunk_position in POSITION_ORDER:
            group = groups.get(chunk_position)
            if not group:
                continue
            all_results.extend(_search_position_group(store.collection, chunk_position, group))

        deduped: dict[tuple, dict] = {}
        for item in all_results:
            key = (item.get("filename"), item.get("chunk_index"))
            deduped.setdefault(key, item)

        retrieved_chunks = [
            {
                "content": item["content"],
                "category": item["category"],
                "score": item["score"],
                "chunk_position": item["chunk_position"],
            }
            for item in deduped.values()
        ]

        return {"retrieved_chunks": retrieved_chunks}
    except Exception as exc:
        return {"error": f"retriever_node failed: {exc}", "retrieved_chunks": []}
