import logging
import os
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.agent.graph import agent_graph
from app.schemas.inference import InferenceResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/inference", response_model=InferenceResponse)
def run_inference(file: UploadFile = File(...)) -> InferenceResponse:
    """Runs the LangGraph agent to natural completion — no timeout, no page cap here.

    Defined as a plain (non-async) function so FastAPI dispatches it to a worker
    thread automatically, letting this request run as long as it needs without
    blocking the event loop for other requests.
    """
    contents = file.file.read()
    suffix = Path(file.filename).suffix or ".pdf"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(contents)
    tmp.close()

    start = time.time()
    try:
        state = agent_graph.invoke({"file_path": tmp.name, "filename": file.filename})
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    return InferenceResponse(
        filename=file.filename,
        final_category=state.get("final_category", ""),
        final_confidence=state.get("final_confidence", 0.0),
        reasoning=state.get("reasoning", ""),
        fallback_triggered=bool(state.get("fallback_triggered", False)),
        vote_breakdown=state.get("vote_counts", {}),
        processing_time_seconds=round(time.time() - start, 2),
        error=state.get("error"),
    )
