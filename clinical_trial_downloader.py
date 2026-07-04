"""
Clinical Trial PDF Bulk Downloader
====================================
Downloads Protocol, SAP, and ICF PDFs from ClinicalTrials.gov API
Target: 50-60 labeled PDF files organized by document type

Usage:
    pip install requests
    python download_clinical_trials.py
"""

import os
import time
import requests
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG — tweak these if needed
# ─────────────────────────────────────────────
OUTPUT_DIR = "clinical_trial_pdfs"   # folder where PDFs will be saved
SLEEP_BETWEEN_REQUESTS = 1.0         # seconds between API calls (be polite)
SLEEP_BETWEEN_DOWNLOADS = 1.5        # seconds between PDF downloads
MAX_PAGES_PER_CONDITION = 15         # safety cap on pagination depth per condition

# Desired FINAL total per category (script tops up from what's already on disk)
CATEGORY_TARGETS = {
    "Protocol": 35,
    "SAP": 28,
    "ICF": 36,
    "Combined": 26,
    "Other": 15,   # any doc type that doesn't match Protocol/SAP/ICF/Combined
}

# Conditions to search — ordered by likelihood of having PDFs
SEARCH_CONDITIONS = [
    "breast cancer",
    "lung cancer",
    "type 2 diabetes",
    "heart failure",
    "Alzheimer disease",
    "colorectal cancer",
    "atrial fibrillation",
    "Parkinson disease",
    "prostate cancer",
    "ovarian cancer",
]

# ─────────────────────────────────────────────
# SETUP FOLDERS
# ─────────────────────────────────────────────
def setup_folders(base_dir):
    folders = {
        "Protocol":  os.path.join(base_dir, "01_Protocol"),
        "SAP":       os.path.join(base_dir, "02_SAP"),
        "ICF":       os.path.join(base_dir, "03_ICF"),
        "Combined":  os.path.join(base_dir, "04_Combined"),
        "Other":     os.path.join(base_dir, "05_Other"),
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    return folders


# ─────────────────────────────────────────────
# COUNT EXISTING PDFs PER CATEGORY
# ─────────────────────────────────────────────
def count_existing(folders: dict) -> dict:
    counts = {}
    for label, folder in folders.items():
        counts[label] = len([f for f in os.listdir(folder) if f.lower().endswith(".pdf")])
    return counts


# ─────────────────────────────────────────────
# DETERMINE DOCUMENT TYPE FROM FILENAME
# ─────────────────────────────────────────────
def get_doc_type(filename: str, folders: dict) -> tuple:
    """Returns (label, folder_path) based on filename prefix."""
    fn = filename.upper()
    if fn.startswith("PROT_SAP_ICF") or fn.startswith("PROT_ICF") or fn.startswith("PROT_SAP"):
        return "Combined", folders["Combined"]
    elif fn.startswith("PROT"):
        return "Protocol", folders["Protocol"]
    elif fn.startswith("SAP"):
        return "SAP", folders["SAP"]
    elif fn.startswith("ICF"):
        return "ICF", folders["ICF"]
    else:
        return "Other", folders["Other"]  # unrecognized doc type


# ─────────────────────────────────────────────
# FETCH STUDIES FROM API
# ─────────────────────────────────────────────
def fetch_studies_with_docs(condition: str, max_studies: int = 50, page_token: str = None) -> tuple:
    """
    Calls ClinicalTrials.gov API v2 to find studies with uploaded documents.
    Returns (studies, next_page_token).
    """
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.cond": condition,
        "filter.advanced": "AREA[HasResults]false",   # include trials w/o results too
        "aggFilters": "docs:prot",                    # only studies with Protocol docs
        "fields": "NCTId,BriefTitle,LargeDocumentModule",
        "pageSize": max_studies,
        "format": "json",
    }
    if page_token:
        params["pageToken"] = page_token

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        studies = data.get("studies", [])
        print(f"  Found {len(studies)} studies for '{condition}'")
        return studies, data.get("nextPageToken")
    except Exception as e:
        print(f"  ⚠ API error for '{condition}': {e}")
        return [], None


# ─────────────────────────────────────────────
# EXTRACT PDF LINKS FROM STUDY
# ─────────────────────────────────────────────
def extract_pdf_links(study: dict) -> list:
    """
    Extracts downloadable PDF info from a study's LargeDocSection.
    Returns list of dicts: {nct_id, filename, label, url}
    """
    pdfs = []

    try:
        nct_id = study["protocolSection"]["identificationModule"]["nctId"]
    except KeyError:
        return pdfs

    # Navigate to large docs
    large_doc_section = study.get("documentSection", {}).get("largeDocumentModule", {})
    large_docs = large_doc_section.get("largeDocs", [])

    for doc in large_docs:
        filename = doc.get("filename", "")
        label = doc.get("label", "")

        if not filename.endswith(".pdf"):
            continue

        # Build CDN URL — pattern: last 2 digits of NCT number as folder
        nct_suffix = nct_id[-2:]  # e.g. NCT02107703 → "03"
        pdf_url = f"https://cdn.clinicaltrials.gov/large-docs/{nct_suffix}/{nct_id}/{filename}"

        pdfs.append({
            "nct_id": nct_id,
            "filename": filename,
            "label": label,
            "url": pdf_url,
        })

    return pdfs


# ─────────────────────────────────────────────
# DOWNLOAD A SINGLE PDF
# ─────────────────────────────────────────────
def download_pdf(pdf_info: dict, folders: dict) -> bool:
    """Downloads one PDF to the appropriate folder. Returns True on success."""
    url = pdf_info["url"]
    nct_id = pdf_info["nct_id"]
    filename = pdf_info["filename"]

    doc_type, folder = get_doc_type(filename, folders)

    # Prefix filename with NCT ID to avoid collisions
    save_name = f"{nct_id}_{filename}"
    save_path = os.path.join(folder, save_name)

    # Skip if already downloaded
    if os.path.exists(save_path):
        print(f"  ↷ Already exists: {save_name}")
        return False

    try:
        headers = {"User-Agent": "Mozilla/5.0 (research project)"}
        resp = requests.get(url, headers=headers, timeout=60, stream=True)

        if resp.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_kb = os.path.getsize(save_path) // 1024
            print(f"  ✓ [{doc_type}] {save_name} ({size_kb} KB)")
            return True
        else:
            print(f"  ✗ HTTP {resp.status_code} for {url}")
            return False

    except Exception as e:
        print(f"  ✗ Download error: {e}")
        return False


# ─────────────────────────────────────────────
# WRITE SUMMARY LOG
# ─────────────────────────────────────────────
def write_summary(base_dir: str, downloaded: list):
    log_path = os.path.join(base_dir, "download_log.txt")
    with open(log_path, "w") as f:
        f.write("Clinical Trial PDF Download Log\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total PDFs downloaded: {len(downloaded)}\n\n")

        # Count by type
        counts = {cat: 0 for cat in CATEGORY_TARGETS}
        for item in downloaded:
            counts[item["doc_type"]] = counts.get(item["doc_type"], 0) + 1

        f.write("By document type:\n")
        for dtype, count in counts.items():
            f.write(f"  {dtype}: {count}\n")

        f.write("\nFile list:\n")
        f.write("-" * 60 + "\n")
        for item in downloaded:
            f.write(f"[{item['doc_type']}] {item['nct_id']} | {item['filename']} | {item['label']}\n")

    print(f"\n📄 Summary log saved: {log_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Clinical Trial PDF Bulk Downloader")
    print("=" * 60)

    folders = setup_folders(OUTPUT_DIR)
    existing = count_existing(folders)
    remaining = {cat: max(0, target - existing.get(cat, 0)) for cat, target in CATEGORY_TARGETS.items()}

    print(f"\n📁 Output folder: {os.path.abspath(OUTPUT_DIR)}")
    print("🎯 Targets (existing → target, still needed):")
    for cat, target in CATEGORY_TARGETS.items():
        print(f"   {cat:10s}: {existing.get(cat, 0)} → {target}  (need {remaining[cat]} more)")
    print()

    downloaded = []
    seen_ncts = set()  # avoid duplicate studies across conditions

    def all_satisfied():
        return all(v <= 0 for v in remaining.values())

    for condition in SEARCH_CONDITIONS:
        if all_satisfied():
            break

        print(f"\n🔍 Searching: '{condition}'")
        page_token = None
        for _ in range(MAX_PAGES_PER_CONDITION):
            if all_satisfied():
                break

            studies, page_token = fetch_studies_with_docs(condition, max_studies=30, page_token=page_token)
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            if not studies:
                break

            for study in studies:
                if all_satisfied():
                    break

                try:
                    nct_id = study["protocolSection"]["identificationModule"]["nctId"]
                except KeyError:
                    continue

                if nct_id in seen_ncts:
                    continue
                seen_ncts.add(nct_id)

                pdf_links = extract_pdf_links(study)
                if not pdf_links:
                    continue

                for pdf_info in pdf_links:
                    if all_satisfied():
                        break

                    doc_type, _ = get_doc_type(pdf_info["filename"], folders)
                    if remaining.get(doc_type, 0) <= 0:
                        continue  # already have enough of this category

                    print(f"\n  📋 {nct_id} — {pdf_info['filename']}")
                    success = download_pdf(pdf_info, folders)
                    if success:
                        remaining[doc_type] -= 1
                        downloaded.append({
                            "nct_id": pdf_info["nct_id"],
                            "filename": pdf_info["filename"],
                            "label": pdf_info["label"],
                            "doc_type": doc_type,
                        })

                    time.sleep(SLEEP_BETWEEN_DOWNLOADS)

            if not page_token:
                break

    # Final summary
    print("\n" + "=" * 60)
    print(f"✅ Download complete! New PDFs downloaded: {len(downloaded)}")
    for cat in CATEGORY_TARGETS:
        got = sum(1 for d in downloaded if d["doc_type"] == cat)
        print(f"   {cat:10s}: +{got}  (still short by {remaining[cat]})" if remaining[cat] > 0 else f"   {cat:10s}: +{got}  (target met)")
    print("=" * 60)

    write_summary(OUTPUT_DIR, downloaded)
    print(f"\n📂 All files saved in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()