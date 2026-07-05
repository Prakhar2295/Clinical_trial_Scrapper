import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.categories import router as categories_router
from app.api.routes.health import router as health_router
from app.api.routes.inference import router as inference_router
from app.api.routes.ingest import router as ingest_router
from app.core.config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Clinical Trial Document Tagger API",
    description="AI-powered clinical trial document classification. Endpoints: /health, /ingest, /inference, /categories",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ingest_router)
app.include_router(inference_router)
app.include_router(health_router)
app.include_router(categories_router)
