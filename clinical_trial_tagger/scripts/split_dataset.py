import math
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent / "clinical_trial_pdfs"

CATEGORIES = ["01_Protocol", "02_SAP", "03_ICF", "04_Combined"]

print("=" * 60)
print("DATASET SPLIT (75% ingestion / 25% inference)")
print("=" * 60)

summary = []
for cat_dir in CATEGORIES:
    folder = BASE / cat_dir
    # Top-level PDFs only — ignore any nested subdirectories.
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    total = len(files)

    n_ingest = max(1, math.floor(total * 0.75)) if total > 1 else total
    n_infer = total - n_ingest
    if n_infer == 0 and total > 1:
        # Ensure at least 1 file lands in inference/ even for small folders.
        n_ingest -= 1
        n_infer = 1

    ingest_dir = folder / "ingestion"
    infer_dir = folder / "inference"
    ingest_dir.mkdir(exist_ok=True)
    infer_dir.mkdir(exist_ok=True)

    for f in files[:n_ingest]:
        shutil.move(str(f), str(ingest_dir / f.name))
    for f in files[n_ingest:]:
        shutil.move(str(f), str(infer_dir / f.name))

    print(f"{cat_dir:12s}: total={total:3d}  ingestion={n_ingest:3d}  inference={n_infer:3d}")
    summary.append((cat_dir, total, n_ingest, n_infer))

print("=" * 60)
total_all = sum(s[1] for s in summary)
ingest_all = sum(s[2] for s in summary)
infer_all = sum(s[3] for s in summary)
print(f"{'TOTAL':12s}: total={total_all:3d}  ingestion={ingest_all:3d}  inference={infer_all:3d}")
print("=" * 60)
