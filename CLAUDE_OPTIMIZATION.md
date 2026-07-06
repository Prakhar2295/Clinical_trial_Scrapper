# CLAUDE_OPTIMIZATION.md — Clinical Trial Tagger: Accuracy Optimization Phase

## Context

The base pipeline is fully built and working end-to-end:
- Docling PDF extraction → TOC-aware MarkdownTextSplitter chunking
- Jina v3 embeddings (1024-dim, task="retrieval.passage")
- Weaviate embedded hybrid search (native BM25 + dense, no Rerank)
- LangGraph agent (extractor → retriever → classifier → deep_reader → finalizer)
- Claude Haiku as LLM judge in finalizer_node
- FastAPI with 5 endpoints: /health, /ingest, /inference, /categories, /collections/manage
- chunks_debug/ and tagged_files/ for output inspection

## Goal of This Phase

Achieve at least 75-80% classification accuracy per category through
systematic evaluation and iterative optimization. Do NOT rewrite the
existing pipeline — iterate and improve it.

---

## Document Folders

PDF files are located at:
```
../clinical_trial_pdfs/
├── 01_Protocol/    ← Protocol documents
├── 02_SAP/         ← Statistical Analysis Plan documents
├── 03_ICF/         ← Informed Consent Form documents
├── 04_Combined/    ← Combined document types
└── 05_Other/       ← IGNORE this folder entirely
```

**Ignore 05_Other completely** — do not ingest or test any files from it.

---

## Phase 1 — Data Split and Ingestion

### Step 1 — Purge Weaviate completely
Call the purge endpoint before doing anything else:
```bash
curl -X POST http://localhost:8000/collections/manage \
  -H "Content-Type: application/json" \
  -H "X-Confirm-Purge: yes" \
  -d '{"action": "purge_all"}'
```
Confirm response shows deleted_chunks > 0 or 0 if already empty.
Also delete contents of chunks_debug/ and tagged_files/ directories.

### Step 2 — Count files per category
Scan each folder and count total PDFs:
```
01_Protocol/ → count files → N_protocol
02_SAP/      → count files → N_sap
03_ICF/      → count files → N_icf
04_Combined/ → count files → N_combined
```
Print a summary table before splitting.

### Step 3 — Split 75% ingest / 25% inference per category
For each category folder:
- Sort files alphabetically for reproducibility
- Take first 75% (floor) → move to {folder}/ingestion/
- Take remaining 25% (ceil) → move to {folder}/inference/

Example with 20 files:
```
15 files → 01_Protocol/ingestion/
5 files  → 01_Protocol/inference/
```

Minimum 1 file must exist in inference/ even if folder is tiny.
Print exact split counts per category before proceeding.

### Step 4 — Ingest all files from ingestion/ subfolders
Use curl to POST each file to /ingest with correct category label:

```bash
# For each file in 01_Protocol/ingestion/:
curl -X POST http://localhost:8000/ingest \
  -F "file=@{filepath}" \
  -F "category=Protocol"

# For each file in 02_SAP/ingestion/:
curl -X POST http://localhost:8000/ingest \
  -F "file=@{filepath}" \
  -F "category=SAP"

# For each file in 03_ICF/ingestion/:
curl -X POST http://localhost:8000/ingest \
  -F "file=@{filepath}" \
  -F "category=ICF"

# For each file in 04_Combined/ingestion/:
curl -X POST http://localhost:8000/ingest \
  -F "file=@{filepath}" \
  -F "category=Combined"
```

Wait 30 seconds after each file upload before the next one —
background ingestion needs time to complete (Docling + embedding).
Print progress: "Ingesting [Protocol] filename.pdf (3/15)..."

After all files submitted, wait 60 seconds then call:
```bash
curl -X POST http://localhost:8000/collections/manage \
  -H "Content-Type: application/json" \
  -d '{"action": "stats"}'
```
Print the stats. Confirm chunk counts look reasonable before testing.

---

## Phase 2 — Evaluation

### Step 5 — Run inference on all inference/ files
For each file in each category's inference/ folder:
- POST to /inference endpoint
- Record: filename, true_category, predicted_category,
  final_confidence, fallback_triggered, vote_breakdown,
  processing_time_seconds, error

Save all results to scripts/eval_results.json:
```json
[
  {
    "filename": "NCT02005770_Prot_000.pdf",
    "true_category": "Protocol",
    "predicted_category": "Protocol",
    "correct": true,
    "final_confidence": 0.91,
    "fallback_triggered": false,
    "vote_breakdown": {"Protocol": 8, "SAP": 2},
    "processing_time_seconds": 18.4,
    "error": null
  }
]
```

### Step 6 — Calculate accuracy metrics
After all inference runs complete, calculate:

Per-category accuracy:
```
Protocol accuracy  = correct Protocol predictions / total Protocol files
SAP accuracy       = correct SAP predictions / total SAP files
ICF accuracy       = correct ICF predictions / total ICF files
Combined accuracy  = correct Combined predictions / total Combined files
Overall accuracy   = total correct / total files
```

Also calculate confusion matrix — what does each category get
misclassified as:
```
True\Predicted  Protocol  SAP  ICF  Combined
Protocol           X       2    0      1
SAP                1       X    0      0
ICF                0       0    X      1
Combined           2       1    0      X
```

Save full report to scripts/eval_report.json.

### Step 7 — Print evaluation summary
Print a clear ASCII table:
```
========================================
EVALUATION RESULTS
========================================
Category    Files  Correct  Accuracy
----------------------------------------
Protocol      5      4       80.0%
SAP           5      3       60.0%
ICF           4      3       75.0%
Combined      3      2       66.7%
----------------------------------------
OVERALL      17     12       70.6%
========================================

CONFUSION MATRIX:
...

LOW CONFIDENCE predictions (< 0.7):
  filename.pdf → predicted SAP (0.52) | true: Protocol

FALLBACK TRIGGERED:
  filename.pdf → fallback used, predicted ICF
========================================
```

---

## Phase 3 — Bottleneck Identification (LLM as Judge)

### Step 8 — Identify failures
For every incorrect prediction:
- Load the chunks.txt from chunks_debug/{filename}/
- Load the inference response
- Use Claude Haiku as judge with this prompt:

```
You are an expert evaluator of a clinical trial document 
classification system.

TRUE CATEGORY: {true_category}
PREDICTED CATEGORY: {predicted_category}
CONFIDENCE: {final_confidence}
VOTE BREAKDOWN: {vote_breakdown}
FALLBACK TRIGGERED: {fallback_triggered}

DOCUMENT CHUNKS THAT WERE INGESTED:
{chunks_content}

RETRIEVED CHUNKS USED FOR CLASSIFICATION:
{retrieved_chunks}

REASONING THE CLASSIFIER GAVE:
{reasoning}

Analyze why this document was misclassified. Identify:
1. Was the chunking capturing the right signals? 
   (title page, key sections, strong category-specific language)
2. Were the retrieved chunks from the correct category?
   Or did wrong-category chunks dominate the vote?
3. Was the LLM reasoning correct but overridden by bad votes?
   Or was the LLM itself confused?
4. What specific text signals were MISSING from the chunks
   that would have identified the correct category?
5. What is the PRIMARY bottleneck causing this misclassification?
   Choose one: CHUNKING | RETRIEVAL | VOTING | LLM_REASONING | CORPUS_SIZE

Respond ONLY with JSON:
{
  "primary_bottleneck": "CHUNKING|RETRIEVAL|VOTING|LLM_REASONING|CORPUS_SIZE",
  "chunking_issue": "description or null",
  "retrieval_issue": "description or null", 
  "voting_issue": "description or null",
  "llm_issue": "description or null",
  "missing_signals": ["signal1", "signal2"],
  "recommendation": "specific actionable fix"
}
```

Save all judge outputs to scripts/bottleneck_analysis.json.

### Step 9 — Aggregate bottlenecks
Count bottleneck types across all failures:
```
CHUNKING:      3 failures
RETRIEVAL:     5 failures
VOTING:        2 failures
LLM_REASONING: 1 failure
CORPUS_SIZE:   4 failures
```
The most common bottleneck is attacked first.

---

## Phase 4 — Optimization Iterations

Work through these optimization strategies in order based on
bottleneck analysis. Implement one at a time, re-evaluate,
measure accuracy improvement before moving to the next.

### Optimization Strategy 1 — Prompt Engineering (if LLM_REASONING bottleneck)

Update SYSTEM_PROMPT in app/agent/nodes/finalizer.py with:
- More explicit category definitions with distinguishing features
- Few-shot examples showing correct classification with reasoning
- Explicit instructions on what makes each category unique

Category definitions to add to prompt:
```
Protocol: Contains study objectives, inclusion/exclusion criteria,
  study design, dose/treatment regimens, primary/secondary endpoints.
  Keywords: "study protocol", "primary endpoint", "phase I/II/III",
  "randomization", "inclusion criteria", "investigational product"

SAP: Contains statistical methodology, analysis populations,
  hypothesis testing, sample size calculations.
  Keywords: "statistical analysis plan", "primary analysis",
  "ANCOVA", "intent-to-treat", "multiplicity", "power calculation"

ICF: Contains voluntary participation language, risks/benefits,
  withdrawal rights, confidentiality, signature lines.
  Keywords: "informed consent", "voluntary", "right to withdraw",
  "risks and benefits", "confidentiality", "signature"

Combined: Contains elements of multiple document types above.
  Look for mixed signals — protocol sections AND statistical
  sections AND consent sections all present.
```

Add 2-3 few-shot examples per category directly in the prompt.
Re-run evaluation after this change. Measure accuracy delta.

### Optimization Strategy 2 — External Reranker (if RETRIEVAL bottleneck)

If retrieval is returning wrong-category chunks at top positions,
add a lightweight cross-encoder reranker client-side in
retriever_node.py AFTER Weaviate returns results:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(query: str, results: list[dict]) -> list[dict]:
    pairs = [(query, r["content"]) for r in results]
    scores = reranker.predict(pairs)
    for i, result in enumerate(results):
        result["rerank_score"] = float(scores[i])
    return sorted(results, key=lambda x: x["rerank_score"], 
                  reverse=True)
```

Add cross-encoder/ms-marco-MiniLM-L-6-v2 back to requirements.txt.
Apply reranker after hybrid search returns, before passing to
classifier_node. Re-run evaluation. Measure accuracy delta.

### Optimization Strategy 3 — Long Context Retrieval (if CHUNKING bottleneck)

If chunks are missing critical signals because they cut mid-section:

Option A — Increase chunk size:
In app/core/chunker.py, increase chunk_size from 8000 to 12000
characters (~3000 tokens, still within Jina v3's 8192 limit).
Purge Weaviate, re-ingest, re-evaluate.

Option B — Add section header extraction:
In app/core/extractor.py, after Docling extraction, extract all
markdown headings (## and ###) and create a separate
"document outline" chunk tagged chunk_position="outline".
This gives the retriever a structured signal about what sections
exist in the document even if full content is chunked differently.
Purge Weaviate, re-ingest, re-evaluate.

Option C — Parent-child chunking:
Store full page as parent chunk (for context) but embed smaller
512-token child chunks (for precision). Retrieve by child,
return parent content to classifier. This is the most complex
option — only implement if A and B don't work.

### Optimization Strategy 4 — Change Embedding Model (if all else fails)

If accuracy is still below 70% after strategies 1-3, switch
embedding model. Try in this order:

1. BAAI/bge-large-en-v1.5 (512 token window, 768-dim, MTEB 64)
2. intfloat/e5-large-v2 (512 token window, 1024-dim, MTEB 62)
3. sentence-transformers/all-mpnet-base-v2 (384-dim, very fast)

For each switch:
- Update EMBEDDING_MODEL and EMBEDDING_DIMENSIONS in config.py
- Update Weaviate schema dimensions
- Purge Weaviate completely
- Re-ingest all ingestion/ files
- Re-run full evaluation
- Compare accuracy to previous model

### Optimization Strategy 5 — Corpus Balance (if CORPUS_SIZE bottleneck)

If a category has fewer than 10 ingestion files:
- It will always underperform due to thin retrieval evidence
- Download more files for that category using the existing
  download_clinical_trials.py script
- Re-ingest the additional files
- Re-evaluate

Do not change the 75/25 split ratio — just increase total files.

---

## Phase 5 — Iteration Loop

After each optimization:
1. Re-run evaluation (Phase 2 steps 5-7)
2. Re-run bottleneck analysis (Phase 3 steps 8-9)
3. Check if accuracy improved
4. If yes and >= 75% per category → DONE
5. If yes but < 75% → continue to next optimization
6. If no improvement → revert the change, try next strategy

Keep a running log in scripts/optimization_log.json:
```json
[
  {
    "iteration": 1,
    "change": "Added few-shot examples to finalizer prompt",
    "accuracy_before": {"Protocol": 0.60, "SAP": 0.60, 
                        "ICF": 0.75, "Combined": 0.67,
                        "overall": 0.65},
    "accuracy_after":  {"Protocol": 0.80, "SAP": 0.70,
                        "ICF": 0.75, "Combined": 0.67,
                        "overall": 0.73},
    "delta": "+0.08 overall",
    "decision": "KEEP — significant improvement"
  }
]
```

---

## Target Accuracy

```
Minimum acceptable per category:  75%
Target per category:               80%+
Overall minimum:                   75%
```

Do not stop iterating until these targets are met or all 5
optimization strategies have been exhausted. If all strategies
are exhausted and targets are not met, produce a final report
explaining which bottleneck is hardest to overcome and what
additional data or infrastructure would be needed.

---

## Implementation Order

Execute in EXACTLY this order. Do not skip steps.
Do not proceed to the next step until the current one is complete
and verified.

```
Step 1  → Purge Weaviate + clean directories
Step 2  → Count files per category
Step 3  → Split 75/25 into ingestion/ and inference/ subfolders
Step 4  → Ingest all ingestion/ files via curl (wait between each)
Step 5  → Run inference on all inference/ files, save eval_results.json
Step 6  → Calculate accuracy metrics + confusion matrix
Step 7  → Print evaluation summary
Step 8  → Run LLM-as-judge on all failures, save bottleneck_analysis.json
Step 9  → Aggregate bottlenecks, identify primary bottleneck
Step 10 → Apply optimization strategy based on bottleneck
Step 11 → Re-evaluate, measure delta
Step 12 → Repeat from Step 5 until accuracy targets met
```

---

## Key Constraints

- Server must be running at http://localhost:8000 throughout
- Never modify the 75/25 split once made
- Never move files back from inference/ to ingestion/
- Always purge Weaviate before changing embedding model or chunk size
- LLM judge calls use Claude Haiku only (cheap, fast)
- Save ALL intermediate results to scripts/ — never discard data
- If a curl ingest call fails, log it and continue — do not crash
- Print clear progress updates at every step
- The optimization_log.json must be updated after every iteration
- Do not rewrite the core pipeline — only tune parameters,
  prompts, and strategies within the existing architecture
```