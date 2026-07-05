import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHUNKS_DEBUG_DIR = BASE_DIR / "chunks_debug"

SEPARATOR = "=" * 80


def write_chunk_debug(
    filename: str,
    category: str,
    ingested_at: str,
    total_pages: int,
    chunks: list[dict],
    source_type: str,
) -> None:
    """Writes per-document chunk debug output for manual QA of the ingestion chunker."""
    out_dir = CHUNKS_DEBUG_DIR / Path(filename).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered_chunks = sorted(chunks, key=lambda c: c["chunk_index"])

    metadata = {
        "filename": filename,
        "category": category,
        "ingested_at": ingested_at,
        "total_pages": total_pages,
        "total_chunks": len(ordered_chunks),
        "head_chunks": sum(1 for c in ordered_chunks if c.get("chunk_position") == "head"),
        "tail_chunks": sum(1 for c in ordered_chunks if c.get("chunk_position") == "tail"),
        "source_type": source_type,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    blocks = []
    for chunk in ordered_chunks:
        content = chunk["content"]
        blocks.append(
            f"{SEPARATOR}\n"
            f"CHUNK {chunk['chunk_index']:02d} | POSITION: {chunk['chunk_position']} "
            f"| PAGE RANGE: {chunk['page_range']} | CHARS: {len(content)}\n"
            f"{SEPARATOR}\n\n"
            f"{content}\n"
        )
    blocks.append(f"{SEPARATOR}\nEND OF FILE — {filename} | TOTAL CHUNKS: {len(ordered_chunks)}\n{SEPARATOR}\n")

    (out_dir / "chunks.txt").write_text("\n".join(blocks))
