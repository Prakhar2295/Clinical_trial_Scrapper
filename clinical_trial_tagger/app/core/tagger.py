import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TAGGED_FILES_DIR = BASE_DIR / "tagged_files"


def write_tagged_file(
    original_file_path: str,
    filename: str,
    final_category: str,
    final_confidence: float,
    reasoning: str,
    vote_breakdown: dict,
    fallback_triggered: bool,
    classified_at: str,
) -> None:
    """Copies the classified PDF into tagged_files/ with its category in the filename,
    alongside a metadata JSON sidecar."""
    TAGGED_FILES_DIR.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".pdf"
    tagged_filename = f"{stem}_[{final_category}]{suffix}"

    shutil.copy2(original_file_path, TAGGED_FILES_DIR / tagged_filename)

    nct_id = filename.split("_", 1)[0] if "_" in filename else stem

    metadata = {
        "filename": filename,
        "nct_id": nct_id,
        "tagged_filename": tagged_filename,
        "final_category": final_category,
        "final_confidence": final_confidence,
        "reasoning": reasoning,
        "vote_breakdown": vote_breakdown,
        "fallback_triggered": fallback_triggered,
        "classified_at": classified_at,
        "source_type": "inference",
    }

    meta_filename = f"{Path(tagged_filename).stem}_meta.json"
    (TAGGED_FILES_DIR / meta_filename).write_text(json.dumps(metadata, indent=2))
