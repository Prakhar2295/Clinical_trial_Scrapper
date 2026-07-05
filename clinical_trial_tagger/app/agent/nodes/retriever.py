from collections import defaultdict

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


def _search_position_group(store, chunk_position: str, group: list[dict]) -> list[dict]:
    chunk_text = "\n".join(c["content"] for c in group)[:2000]
    embedding = _embedder.embed_one(chunk_text).tolist()

    return store.query_hybrid(
        query_text=chunk_text,
        vector=embedding,
        chunk_position=chunk_position,
        limit=settings.top_k_retrieval,
    )


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
            all_results.extend(_search_position_group(store, chunk_position, group))

        deduped: dict[tuple, dict] = {}
        for item in all_results:
            key = (item.get("filename"), item.get("content"))
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
