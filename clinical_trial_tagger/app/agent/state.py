from typing import Optional, TypedDict


class AgentState(TypedDict):
    # Input
    file_path: str
    filename: str

    # Extraction outputs
    extracted_text: str  # markdown from Docling
    chunks: list[dict]  # [{content, chunk_position, chunk_index, page_range}]

    # Retrieval outputs
    retrieved_chunks: list[dict]  # [{content, category, score, chunk_position}]

    # Classification
    vote_counts: dict  # {category: count}
    confidence: float
    predicted_category: str

    # Fallback flag
    fallback_triggered: bool
    fallback_pages_read: int

    # Final output
    final_category: str
    final_confidence: float
    reasoning: str
    evidence_chunks: list[str]
    error: Optional[str]
