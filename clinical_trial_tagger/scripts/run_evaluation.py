import json
from collections import defaultdict
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent.parent.parent / "clinical_trial_pdfs"
API_URL = "http://localhost:8000/inference"

CATEGORY_MAP = {
    "01_Protocol": "Protocol",
    "02_SAP": "SAP",
    "03_ICF": "ICF",
    "04_Combined": "Combined",
}
CATEGORIES = list(CATEGORY_MAP.values())

jobs = []
for folder_name, category in CATEGORY_MAP.items():
    infer_dir = BASE / folder_name / "inference"
    for f in sorted(infer_dir.iterdir()):
        if f.is_file() and f.suffix.lower() == ".pdf":
            jobs.append((category, f))

total = len(jobs)
print(f"Total files to run inference on: {total}")

results = []
for i, (true_category, filepath) in enumerate(jobs, start=1):
    print(f"Running inference [{true_category}] {filepath.name} ({i}/{total})...", flush=True)
    try:
        with open(filepath, "rb") as fh:
            resp = requests.post(
                API_URL,
                files={"file": (filepath.name, fh, "application/pdf")},
                timeout=180,
            )
        body = resp.json()
        predicted = body.get("final_category", "")
        result = {
            "filename": filepath.name,
            "true_category": true_category,
            "predicted_category": predicted,
            "correct": predicted == true_category,
            "final_confidence": body.get("final_confidence"),
            "fallback_triggered": body.get("fallback_triggered"),
            "vote_breakdown": body.get("vote_breakdown"),
            "processing_time_seconds": body.get("processing_time_seconds"),
            "error": body.get("error"),
        }
        print(
            f"  -> predicted={predicted} correct={result['correct']} "
            f"confidence={result['final_confidence']}",
            flush=True,
        )
    except Exception as e:
        result = {
            "filename": filepath.name,
            "true_category": true_category,
            "predicted_category": None,
            "correct": False,
            "final_confidence": None,
            "fallback_triggered": None,
            "vote_breakdown": None,
            "processing_time_seconds": None,
            "error": str(e),
        }
        print(f"  -> FAILED: {e}", flush=True)
    results.append(result)

eval_results_path = Path(__file__).resolve().parent / "eval_results.json"
eval_results_path.write_text(json.dumps(results, indent=2))
print(f"\nSaved eval_results.json ({len(results)} entries)")

# Step 6 — metrics
per_cat = {c: {"total": 0, "correct": 0} for c in CATEGORIES}
confusion = {c: defaultdict(int) for c in CATEGORIES}
low_confidence = []
fallback_used = []

for r in results:
    tc = r["true_category"]
    pc = r["predicted_category"] or "ERROR"
    per_cat[tc]["total"] += 1
    if r["correct"]:
        per_cat[tc]["correct"] += 1
    confusion[tc][pc] += 1
    if r["final_confidence"] is not None and r["final_confidence"] < 0.7:
        low_confidence.append(r)
    if r["fallback_triggered"]:
        fallback_used.append(r)

overall_total = sum(v["total"] for v in per_cat.values())
overall_correct = sum(v["correct"] for v in per_cat.values())

report = {
    "per_category_accuracy": {
        c: {
            "total": per_cat[c]["total"],
            "correct": per_cat[c]["correct"],
            "accuracy": (per_cat[c]["correct"] / per_cat[c]["total"]) if per_cat[c]["total"] else None,
        }
        for c in CATEGORIES
    },
    "overall_accuracy": (overall_correct / overall_total) if overall_total else None,
    "overall_total": overall_total,
    "overall_correct": overall_correct,
    "confusion_matrix": {c: dict(confusion[c]) for c in CATEGORIES},
    "low_confidence_predictions": [
        {
            "filename": r["filename"],
            "predicted": r["predicted_category"],
            "confidence": r["final_confidence"],
            "true": r["true_category"],
        }
        for r in low_confidence
    ],
    "fallback_triggered_predictions": [
        {"filename": r["filename"], "predicted": r["predicted_category"], "true": r["true_category"]}
        for r in fallback_used
    ],
}

eval_report_path = Path(__file__).resolve().parent / "eval_report.json"
eval_report_path.write_text(json.dumps(report, indent=2))
print("Saved eval_report.json")

# Step 7 — print summary
print("\n" + "=" * 40)
print("EVALUATION RESULTS")
print("=" * 40)
print(f"{'Category':10s} {'Files':>6s} {'Correct':>8s} {'Accuracy':>10s}")
print("-" * 40)
for c in CATEGORIES:
    t = per_cat[c]["total"]
    cor = per_cat[c]["correct"]
    acc = (cor / t * 100) if t else 0.0
    print(f"{c:10s} {t:6d} {cor:8d} {acc:9.1f}%")
print("-" * 40)
overall_acc = (overall_correct / overall_total * 100) if overall_total else 0.0
print(f"{'OVERALL':10s} {overall_total:6d} {overall_correct:8d} {overall_acc:9.1f}%")
print("=" * 40)

print("\nCONFUSION MATRIX:")
header = "True\\Predicted".ljust(14) + "".join(c.ljust(10) for c in CATEGORIES + ["ERROR"])
print(header)
for tc in CATEGORIES:
    row = tc.ljust(14)
    for pc in CATEGORIES + ["ERROR"]:
        row += str(confusion[tc].get(pc, 0)).ljust(10)
    print(row)

print("\nLOW CONFIDENCE predictions (< 0.7):")
if low_confidence:
    for r in low_confidence:
        print(
            f"  {r['filename']} -> predicted {r['predicted_category']} "
            f"({r['final_confidence']:.2f}) | true: {r['true_category']}"
        )
else:
    print("  (none)")

print("\nFALLBACK TRIGGERED:")
if fallback_used:
    for r in fallback_used:
        print(f"  {r['filename']} -> fallback used, predicted {r['predicted_category']} | true: {r['true_category']}")
else:
    print("  (none)")
print("=" * 40)
