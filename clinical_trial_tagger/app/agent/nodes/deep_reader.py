import hashlib

from app.agent.state import AgentState
from app.core.chunker import chunk_for_inference
from app.core.config import settings
from app.core.extractor import PDFExtractor

_extractor = PDFExtractor()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deep_reader_node(state: AgentState) -> dict:
    if state.get("fallback_triggered"):
        # Safety net: routing should already prevent a second visit here.
        return {}

    try:
        markdown = _extractor.extract_pages(state["file_path"], max_pages=settings.max_pages_fallback)
        new_chunks = chunk_for_inference(markdown, max_pages=settings.max_pages_fallback)

        existing_chunks = state.get("chunks") or []
        seen_hashes = {_content_hash(c["content"]) for c in existing_chunks}

        merged_chunks = list(existing_chunks)
        for chunk in new_chunks:
            h = _content_hash(chunk["content"])
            if h not in seen_hashes:
                seen_hashes.add(h)
                merged_chunks.append(chunk)

        return {
            "fallback_triggered": True,
            "fallback_pages_read": settings.max_pages_fallback,
            "extracted_text": markdown,
            "chunks": merged_chunks,
        }
    except Exception as exc:
        # fallback_triggered must be set even on failure so the hard limit of
        # one deep_reader_node visit per run holds regardless of outcome.
        return {
            "error": f"deep_reader_node failed for {state.get('filename')}: {exc}",
            "fallback_triggered": True,
        }
