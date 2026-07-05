import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException

from app.core.category_registry import category_registry
from app.core.chunk_writer import CHUNKS_DEBUG_DIR
from app.db.weaviate_client import get_weaviate_store
from app.schemas.collections import CollectionAction, CollectionManageRequest, CollectionManageResponse

router = APIRouter()


def _purge_chunks_debug_contents() -> None:
    if not CHUNKS_DEBUG_DIR.exists():
        return
    for child in CHUNKS_DEBUG_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _delete_chunks_debug_by_category(category: str) -> None:
    if not CHUNKS_DEBUG_DIR.exists():
        return
    for subdir in CHUNKS_DEBUG_DIR.iterdir():
        if not subdir.is_dir():
            continue
        meta_path = subdir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            metadata = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if metadata.get("category") == category:
            shutil.rmtree(subdir)


def _delete_chunks_debug_by_filename(filename: str) -> None:
    target_dir = CHUNKS_DEBUG_DIR / Path(filename).stem
    if target_dir.exists():
        shutil.rmtree(target_dir)


@router.post("/collections/manage", response_model=CollectionManageResponse)
def manage_collection(
    request: CollectionManageRequest,
    x_confirm_purge: str | None = Header(default=None, alias="X-Confirm-Purge"),
) -> CollectionManageResponse:
    # Validate request shape before touching Weaviate at all.
    if request.action == CollectionAction.purge_all and x_confirm_purge != "yes":
        raise HTTPException(
            status_code=400,
            detail="Safety check failed: include header X-Confirm-Purge: yes to confirm purge",
        )

    if request.action == CollectionAction.delete_by_category:
        if request.category is None or request.category not in category_registry.all():
            raise HTTPException(
                status_code=422,
                detail="category is required and must be a known category",
            )

    if request.action == CollectionAction.delete_by_filename and request.filename is None:
        raise HTTPException(
            status_code=422,
            detail="filename is required for delete_by_filename action",
        )

    try:
        store = get_weaviate_store()

        if request.action == CollectionAction.stats:
            stats = store.stats()
            return CollectionManageResponse(
                action=request.action.value,
                status="ok",
                message="Collection statistics retrieved",
                stats=stats,
            )

        if request.action == CollectionAction.list_files:
            files = store.list_files()
            return CollectionManageResponse(
                action=request.action.value,
                status="ok",
                message=f"Found {len(files)} unique file(s)",
                files=files,
            )

        if request.action == CollectionAction.purge_all:
            deleted_chunks = store.purge_all()
            _purge_chunks_debug_contents()
            return CollectionManageResponse(
                action=request.action.value,
                status="ok",
                message=f"Collection purged. {deleted_chunks} chunks deleted.",
                deleted_chunks=deleted_chunks,
            )

        if request.action == CollectionAction.delete_by_category:
            deleted_chunks = store.delete_by_category(request.category)
            _delete_chunks_debug_by_category(request.category)
            return CollectionManageResponse(
                action=request.action.value,
                status="ok",
                message=f"Deleted {deleted_chunks} chunks with category '{request.category}'.",
                deleted_chunks=deleted_chunks,
            )

        if request.action == CollectionAction.delete_by_filename:
            deleted_chunks = store.delete_by_filename(request.filename)
            _delete_chunks_debug_by_filename(request.filename)
            return CollectionManageResponse(
                action=request.action.value,
                status="ok",
                message=f"Deleted {deleted_chunks} chunks with filename '{request.filename}'.",
                deleted_chunks=deleted_chunks,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
