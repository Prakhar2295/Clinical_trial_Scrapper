import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile

from app.core.chunker import chunk_for_ingestion
from app.core.embedder import Embedder
from app.core.extractor import PDFExtractor
from app.db.weaviate_client import get_weaviate_store
from app.schemas.ingest import IngestAcceptedResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_extractor = PDFExtractor()
_embedder = Embedder()


def _run_ingestion(file_path: str, filename: str, category: str, nct_id: str) -> None:
    """Background task: full-document extraction, chunking, embedding, and Weaviate write.

    No page limit and no timeout — runs to natural completion regardless of document size.
    """
    try:
        markdown = _extractor.extract_markdown(file_path)
        chunk_dicts = chunk_for_ingestion(markdown)

        items = []
        for chunk in chunk_dicts:
            vector = _embedder.embed_one(chunk["content"]).tolist()
            items.append(
                {
                    "vector": vector,
                    "properties": {
                        "nct_id": nct_id,
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

        store = get_weaviate_store()
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
    nct_id: str | None = Form(None),
) -> IngestAcceptedResponse:
    contents = await file.read()
    suffix = Path(file.filename).suffix or ".pdf"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(contents)
    tmp.close()

    resolved_nct_id = nct_id or file.filename.split("_")[0]

    background_tasks.add_task(_run_ingestion, tmp.name, file.filename, category, resolved_nct_id)

    return IngestAcceptedResponse(
        status="accepted",
        filename=file.filename,
        message="Ingestion started in background",
    )
