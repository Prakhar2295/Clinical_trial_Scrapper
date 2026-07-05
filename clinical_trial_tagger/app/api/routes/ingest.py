import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.core.category_registry import category_registry
from app.core.chunk_writer import write_chunk_debug
from app.core.chunker import chunk_for_ingestion, split_pages
from app.core.embedder import Embedder
from app.core.extractor import PDFExtractor
from app.db.weaviate_client import get_weaviate_store
from app.schemas.ingest import IngestAcceptedResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_extractor = PDFExtractor()
_embedder = Embedder()


def _run_ingestion(file_path: str, filename: str, category: str) -> None:
    """Background task: full-document extraction, chunking, embedding, and Weaviate write.

    No page limit and no timeout — runs to natural completion regardless of document size.
    """
    try:
        store = get_weaviate_store()

        existing = store.find_by_filename(filename)
        if existing:
            logger.info(f"Skipping {filename} — already ingested with {len(existing)} chunks")
            return

        markdown = _extractor.extract_markdown(file_path)
        chunk_dicts = chunk_for_ingestion(markdown)

        write_chunk_debug(
            filename=filename,
            category=category,
            ingested_at=datetime.now().isoformat(timespec="seconds"),
            total_pages=len(split_pages(markdown)),
            chunks=chunk_dicts,
            source_type="bootstrap",
        )

        items = []
        for chunk in chunk_dicts:
            vector = _embedder.embed_one(chunk["content"]).tolist()
            items.append(
                {
                    "vector": vector,
                    "properties": {
                        "filename": filename,
                        "category": category,
                        "chunk_index": chunk["chunk_index"],
                        "chunk_position": chunk["chunk_position"],
                        "page_range": chunk["page_range"],
                        "content": chunk["content"],
                        "source_type": "bootstrap",
                    },
                }
            )

        store.add_chunks_batch(items)

        logger.info("Ingested %s: %d chunks", filename, len(items))
    except Exception:
        logger.exception("Ingestion failed for %s", filename)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@router.post("/ingest", response_model=IngestAcceptedResponse)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(...),
) -> IngestAcceptedResponse:
    if not category_registry.exists(category):
        raise HTTPException(
            status_code=422,
            detail=f"Unknown category '{category}'. "
            f"Valid categories: {category_registry.all()}. "
            f"Add new categories via POST /categories first.",
        )

    # Canonicalize to the registry's exact stored spelling, regardless of the casing
    # the client submitted, so votes for the same category never split across casings.
    category = next(c for c in category_registry.all() if c.lower() == category.lower())

    contents = await file.read()
    suffix = Path(file.filename).suffix or ".pdf"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(contents)
    tmp.close()

    background_tasks.add_task(_run_ingestion, tmp.name, file.filename, category)

    return IngestAcceptedResponse(
        status="accepted",
        filename=file.filename,
        message="Ingestion started in background",
    )
