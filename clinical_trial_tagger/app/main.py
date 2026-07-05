import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.inference import router as inference_router
from app.api.routes.ingest import router as ingest_router
from app.core.config import settings
from app.db.bm25_store import get_bm25_store
from app.db.weaviate_client import get_weaviate_store

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        store = get_weaviate_store()
        if store.is_ready():
            get_bm25_store().build(store.all_chunks())
    except Exception:
        logger.exception("BM25 rebuild on startup failed; continuing without it")
    yield


app = FastAPI(title="Clinical Trial Document Tagger", lifespan=lifespan)

app.include_router(ingest_router)
app.include_router(inference_router)
app.include_router(health_router)
