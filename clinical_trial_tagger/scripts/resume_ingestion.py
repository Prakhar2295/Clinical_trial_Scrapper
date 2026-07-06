import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent.parent.parent / "clinical_trial_pdfs"
API_URL = "http://localhost:8000/ingest"
COLLECTIONS_URL = "http://localhost:8000/collections/manage"
WAIT_BETWEEN_SECONDS = 15

CATEGORY_MAP = {
    "01_Protocol": "Protocol",
    "02_SAP": "SAP",
    "03_ICF": "ICF",
    "04_Combined": "Combined",
}

# Determine exactly which files remain by diffing ingestion/ folders against
# what's already in Weaviate — don't guess based on position/order.
resp = requests.post(COLLECTIONS_URL, json={"action": "list_files"}, timeout=30)
already_ingested = {f["filename"] for f in resp.json()["files"]}
print(f"Already ingested in Weaviate: {len(already_ingested)} files")

jobs_by_category = {}
for folder_name, category in CATEGORY_MAP.items():
    ingest_dir = BASE / folder_name / "ingestion"
    remaining = sorted(
        p for p in ingest_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf" and p.name not in already_ingested
    )
    if remaining:
        jobs_by_category[category] = remaining

total_remaining = sum(len(v) for v in jobs_by_category.values())
print(f"Remaining to ingest: {total_remaining}")
for cat, files in jobs_by_category.items():
    print(f"  {cat}: {len(files)} remaining")

results = []
job_index = 0
total_jobs = sum(len(v) for v in jobs_by_category.values())
for category, files in jobs_by_category.items():
    n_in_category = len(files)
    for i, filepath in enumerate(files, start=1):
        job_index += 1
        print(f"Submitting [{category}] {filepath.name} ({i}/{n_in_category} remaining)...", flush=True)
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
            results.append(
                {"filename": filepath.name, "category": category, "status_code": None, "ok": False, "error": str(e)}
            )

        if job_index < total_jobs:
            time.sleep(WAIT_BETWEEN_SECONDS)

print("\nAll remaining ingest requests submitted. Waiting 60s for background processing to settle...", flush=True)
time.sleep(60)

failed = [r for r in results if not r["ok"]]
print(f"\nSubmitted: {len(results)}  Accepted (HTTP 200): {len(results) - len(failed)}  Failed to submit: {len(failed)}")
if failed:
    print("Failed submissions:")
    for r in failed:
        print(f"  - {r['category']}/{r['filename']}: {r.get('error', r['status_code'])}")

print("\nFetching final collection stats...")
stats_resp = requests.post(COLLECTIONS_URL, json={"action": "stats"}, timeout=30)
print(stats_resp.json())
