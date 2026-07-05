from app.agent.state import AgentState
from app.core.chunker import chunk_for_inference
from app.core.config import settings
from app.core.extractor import PDFExtractor

_extractor = PDFExtractor()


def extractor_node(state: AgentState) -> dict:
    try:
        markdown = _extractor.extract_pages(state["file_path"], max_pages=settings.max_pages_initial)
        chunk_dicts = chunk_for_inference(markdown, max_pages=settings.max_pages_initial)

        return {
            "extracted_text": markdown,
            "chunks": chunk_dicts,
        }
    except Exception as exc:
        return {"error": f"extractor_node failed for {state.get('filename')}: {exc}"}
