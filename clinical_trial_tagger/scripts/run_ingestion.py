import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent.parent.parent / "clinical_trial_pdfs"
API_URL = "http://localhost:8000/ingest"
WAIT_BETWEEN_SECONDS = 10

CATEGORY_MAP = {
    "01_Protocol": "Protocol",
    "02_SAP": "SAP",
    "03_ICF": "ICF",
    "04_Combined": "Combined",
}

jobs = []
for folder_name, category in CATEGORY_MAP.items():
    ingest_dir = BASE / folder_name / "ingestion"
    for f in sorted(ingest_dir.iterdir()):
        if f.is_file() and f.suffix.lower() == ".pdf":
            jobs.append((category, f))

total = len(jobs)
print(f"Total files to ingest: {total}")

results = []
for i, (category, filepath) in enumerate(jobs, start=1):
    print(f"Ingesting [{category}] {filepath.name} ({i}/{total})...", flush=True)
    try:
        with open(filepath, "rb") as fh:
            resp = requests.post(
                API_URL,
                files={"file": (filepath.name, fh, "application/pdf")},
                data={"category": category},
                timeout=60,
            )
        ok = resp.status_code == 200
        print(f"  -> status={resp.status_code} body={resp.text[:200]}", flush=True)
        results.append({"filename": filepath.name, "category": category, "status_code": resp.status_code, "ok": ok})
    except Exception as e:
        print(f"  -> FAILED: {e}", flush=True)
        results.append({"filename": filepath.name, "category": category, "status_code": None, "ok": False, "error": str(e)})

    if i < total:
        time.sleep(WAIT_BETWEEN_SECONDS)

print("\nAll ingest requests submitted. Waiting 60s for background processing to settle...", flush=True)
time.sleep(60)

failed = [r for r in results if not r["ok"]]
print(f"\nSubmitted: {total}  Accepted (HTTP 200): {total - len(failed)}  Failed to submit: {len(failed)}")
if failed:
    print("Failed submissions:")
    for r in failed:
        print(f"  - {r['category']}/{r['filename']}: {r.get('error', r['status_code'])}")

print("\nFetching final collection stats...")
stats_resp = requests.post(
    "http://localhost:8000/collections/manage",
    json={"action": "stats"},
    timeout=30,
)
print(stats_resp.json())
