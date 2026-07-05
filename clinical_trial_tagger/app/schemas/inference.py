from typing import Optional

from pydantic import BaseModel


class InferenceResponse(BaseModel):
    filename: str
    final_category: str
    final_confidence: float
    reasoning: str
    fallback_triggered: bool
    vote_breakdown: dict
    processing_time_seconds: float
    error: Optional[str] = None
