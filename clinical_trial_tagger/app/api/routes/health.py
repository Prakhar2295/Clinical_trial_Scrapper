import logging

import anthropic
from fastapi import APIRouter

from app.core.config import settings
from app.core.embedder import Embedder
from app.db.weaviate_client import get_weaviate_store

logger = logging.getLogger(__name__)

router = APIRouter()

_embedder = Embedder()


def _check_weaviate() -> str:
    try:
        get_weaviate_store().count()
        return "connected"
    except Exception:
        logger.exception("Weaviate health check failed")
        return "unavailable"


def _check_embedding_model() -> str:
    try:
        _embedder.embed_one("health check")
        return "loaded"
    except Exception:
        logger.exception("Embedding model health check failed")
        return "unavailable"


def _check_anthropic_llm() -> str:
    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        return "connected"
    except Exception:
        logger.exception("Anthropic health check failed")
        return "unavailable"


@router.get("/health")
def health() -> dict:
    weaviate_status = _check_weaviate()
    embedding_status = _check_embedding_model()
    anthropic_status = _check_anthropic_llm()

    all_connected = weaviate_status == "connected" and embedding_status == "loaded" and anthropic_status == "connected"

    return {
        "status": "ok" if all_connected else "degraded",
        "weaviate": weaviate_status,
        "embedding_model": embedding_status,
        "anthropic_llm": anthropic_status,
    }
