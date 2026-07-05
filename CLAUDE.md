# CLAUDE.md — AI-Assisted Clinical Trial Document Tagging Agent

## Project Overview

An end-to-end AI-powered document classification system that automatically tags
clinical trial documents (Protocol, SAP, ICF, CSR, etc.) using a LangGraph
ReAct agent, hybrid vector search via Weaviate, and Anthropic Claude Haiku as
the reasoning LLM. Documents are ingested with metadata labels during a
bootstrap phase, and new untagged documents are classified at inference time via
a FastAPI endpoint.

---

## Tech Stack

| Layer            | Tool                                      |
|------------------|-------------------------------------------|
| PDF Extraction   | Docling                                   |
| Embeddings       | sentence-transformers/all-MiniLM-L6-v2    |
| Sparse Search    | rank_bm25                                 |
| Vector DB        | Weaviate (local Docker)                   |
| Reranker         | cross-encoder/ms-marco-MiniLM-L-6-v2      |
| Agent Framework  | LangGraph (create_react_agent)            |
| LLM              | Anthropic Claude Haiku (claude-haiku-4-5-20251001) |
| API              | FastAPI + Swagger UI                      |
| Config           | python-dotenv (.env file)                 |

---

## Project Structure

Build EXACTLY this folder structure:

```
clinical_trial_tagger/
├── CLAUDE.md
├── .env
├── .env.example
├── requirements.txt
├── docker-compose.yml
├── README.md
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry point
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── classify.py      # POST /classify endpoint
│   │   │   ├── ingest.py        # POST /ingest endpoint
│   │   │   └── feedback.py      # POST /feedback endpoint
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py             # LangGraph ReAct agent definition
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── extractor.py     # Node 1: PDF text + signal extraction
│   │   │   ├── retriever.py     # Node 2: Hybrid search (dense + BM25)
│   │   │   ├── classifier.py    # Node 3: Majority vote + confidence score
│   │   │   ├── deep_reader.py   # Node 4b: Fallback — reads more pages
│   │   │   └── finalizer.py     # Node 5: LLM verify + output
│   │   ├── state.py             # AgentState TypedDict
│   │   └── tools.py             # LangGraph tools (search, keyword scan)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Settings via pydantic-settings
│   │   ├── chunker.py           # Smart page-aware chunking logic
│   │   ├── embedder.py          # SentenceTransformer embedding wrapper
│   │   ├── extractor.py         # Docling PDF → markdown extraction
│   │   ├── keywords.py          # Category keyword signal definitions
│   │   └── reranker.py          # Cross-encoder reranking wrapper
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── weaviate_client.py   # Weaviate connection + schema setup
│   │   └── bm25_store.py        # In-memory BM25 index (rank_bm25)
│   │
│   └── schemas/
│       ├── __init__.py
│       ├── classify.py          # Pydantic request/response models
│       ├── ingest.py
│       └── feedback.py
│
├── scripts/
│   ├── bootstrap_ingest.py      # Bulk ingest labeled PDFs from directory
│   └── test_classify.py         # Quick CLI test of classification pipeline
│
└── tests/
    ├── __init__.py
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_classifier.py
    └── test_api.py
```

---

## Environment Variables

Create `.env` with these keys (also create `.env.example` with empty values):

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=                        # leave empty for local Docker
EMBEDDING_MODEL=all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
TOP_K_RETRIEVAL=10
CONFIDENCE_THRESHOLD=0.85
MAX_PAGES_INITIAL=3                      # pages to extract on first pass
MAX_PAGES_FALLBACK=8                     # pages to extract on deep read
PDF_INPUT_DIR=../clinical_trial_pdfs     # where downloaded PDFs live
LOG_LEVEL=INFO
```

---

## Document Categories

The system classifies into these 6 categories. These are defined in
`app/core/keywords.py` and used as Weaviate metadata:

```python
CATEGORIES = [
    "Protocol",           # Study Protocol
    "SAP",                # Statistical Analysis Plan
    "ICF",                # Informed Consent Form
    "CSR",                # Clinical Study Report
    "IB",                 # Investigator Brochure
    "Combined",           # Docs covering multiple types (Prot_SAP_ICF etc.)
]
```

---

## Keyword Signals

Define in `app/core/keywords.py`. These are used by the Extractor node to
produce a fast keyword-based signal BEFORE embedding search:

```python
CATEGORY_SIGNALS = {
    "Protocol": [
        "study protocol", "primary endpoint", "secondary endpoint",
        "inclusion criteria", "exclusion criteria", "phase i", "phase ii",
        "phase iii", "randomization", "double-blind", "investigational product",
        "dose escalation", "study objectives"
    ],
    "SAP": [
        "statistical analysis plan", "primary analysis", "secondary analysis",
        "multiplicity", "ancova", "mixed model", "intent-to-treat",
        "per protocol", "sensitivity analysis", "sample size calculation",
        "analysis population", "statistical methods"
    ],
    "ICF": [
        "informed consent", "voluntary participation", "right to withdraw",
        "risks and benefits", "confidentiality", "compensation",
        "alternative treatments", "research participant", "hipaa"
    ],
    "CSR": [
        "clinical study report", "study results", "efficacy results",
        "safety results", "adverse events", "serious adverse events",
        "disposition of patients", "protocol deviations", "conclusions"
    ],
    "IB": [
        "investigator brochure", "nonclinical studies", "clinical pharmacology",
        "pharmacokinetics", "pharmacodynamics", "toxicology", "preclinical"
    ],
}
```

---

## Weaviate Schema

Define in `app/db/weaviate_client.py`. Single collection named
`ClinicalTrialChunk`:

```python
SCHEMA = {
    "class": "ClinicalTrialChunk",
    "vectorizer": "none",          # we supply our own vectors
    "properties": [
        {"name": "nct_id",         "dataType": ["text"]},
        {"name": "filename",       "dataType": ["text"]},
        {"name": "category",       "dataType": ["text"]},   # ground truth label
        {"name": "chunk_index",    "dataType": ["int"]},
        {"name": "chunk_position", "dataType": ["text"]},   # "head" | "tail"
        {"name": "page_range",     "dataType": ["text"]},   # e.g. "1-3"
        {"name": "content",        "dataType": ["text"]},   # raw chunk text
        {"name": "keyword_hits",   "dataType": ["text"]},   # JSON list of matched signals
        {"name": "source_type",    "dataType": ["text"]},   # "bootstrap" | "feedback"
    ]
}
```

---

## Chunking Strategy

Implement in `app/core/chunker.py`:

```
For INGESTION (labeled bootstrap docs):
  - Extract first MAX_PAGES_INITIAL pages  → label as chunk_position="head"
  - Extract last 2 pages if doc > 6 pages → label as chunk_position="tail"
  - Chunk each page into 512-token chunks with 64-token overlap
  - Each chunk stored with full metadata

For INFERENCE (new untagged doc — first pass):
  - Extract first 3 pages only → chunk_position="head"
  - If confidence < threshold → fallback: extract up to 8 pages

Chunk size: 512 tokens
Overlap:    64 tokens
Tokenizer:  use tiktoken cl100k_base for counting
```

---

## LangGraph Agent

### State Definition — `app/agent/state.py`

```python
from typing import TypedDict, Optional
from langgraph.graph import MessagesState

class AgentState(TypedDict):
    # Input
    file_path: str
    filename: str

    # Extraction outputs
    extracted_text: str          # markdown from Docling
    keyword_signals: dict        # {category: [matched_keywords]}
    chunks: list[str]            # chunked text list

    # Retrieval outputs
    retrieved_chunks: list[dict] # [{content, category, score, nct_id}]
    bm25_results: list[dict]

    # Classification
    vote_counts: dict            # {category: count}
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
```

### Graph Definition — `app/agent/graph.py`

```
Build a StateGraph with these nodes and edges:

NODES:
  extractor    → app/agent/nodes/extractor.py
  retriever    → app/agent/nodes/retriever.py
  classifier   → app/agent/nodes/classifier.py
  deep_reader  → app/agent/nodes/deep_reader.py
  finalizer    → app/agent/nodes/finalizer.py

EDGES:
  START → extractor
  extractor → retriever
  retriever → classifier
  classifier → [conditional edge]:
      if state.confidence >= CONFIDENCE_THRESHOLD → finalizer
      if state.confidence <  CONFIDENCE_THRESHOLD → deep_reader
  deep_reader → retriever          # re-runs retrieval with more content
  finalizer → END

Use create_react_agent pattern for the finalizer node only
(LLM does final reasoning). All other nodes are deterministic Python.
```

---

## Node Implementations

### Node 1 — Extractor (`app/agent/nodes/extractor.py`)

```
1. Load PDF using Docling DocumentConverter
2. Export to markdown string
3. Split by page breaks, slice first MAX_PAGES_INITIAL pages
4. If fallback_triggered=True, slice up to MAX_PAGES_FALLBACK pages
5. Run keyword signal scan against CATEGORY_SIGNALS dict
   → count hits per category
   → store in state.keyword_signals
6. Run smart chunker on extracted text
7. Store: extracted_text, keyword_signals, chunks in state
```

### Node 2 — Retriever (`app/agent/nodes/retriever.py`)

```
1. Embed each chunk using SentenceTransformer
2. Query Weaviate with dense vector → top TOP_K_RETRIEVAL results
3. Query BM25 index with raw chunk text → top TOP_K_RETRIEVAL results
4. Merge results (reciprocal rank fusion):
   RRF_score = Σ 1/(k + rank_i)  where k=60
5. Take top 10 after fusion
6. Run cross-encoder reranker on merged results
7. Store retrieved_chunks in state
```

### Node 3 — Classifier (`app/agent/nodes/classifier.py`)

```
1. Count category votes from retrieved_chunks metadata
   vote_counts = {"Protocol": 6, "SAP": 3, "ICF": 1}
2. Add keyword signal boost:
   for each category with keyword hits > 2: votes += 2
3. Calculate confidence:
   top_votes = max(vote_counts.values())
   total_votes = sum(vote_counts.values())
   confidence = top_votes / total_votes
4. predicted_category = argmax(vote_counts)
5. Store vote_counts, confidence, predicted_category in state
```

### Node 4b — Deep Reader (`app/agent/nodes/deep_reader.py`)

```
1. Set fallback_triggered = True
2. Re-run Docling extraction up to MAX_PAGES_FALLBACK pages
3. Re-chunk the additional content
4. Merge new chunks with existing chunks (deduplicate by content hash)
5. Update state.chunks with merged set
6. Pass back to retriever node (graph edge handles this)
```

### Node 5 — Finalizer (`app/agent/nodes/finalizer.py`)

```
Call Anthropic Claude Haiku with this prompt structure:

SYSTEM:
  You are an expert clinical trial document classifier.
  Classify documents into one of: Protocol, SAP, ICF, CSR, IB, Combined.
  Always respond with valid JSON only.

USER:
  ## Keyword Signals Detected
  {state.keyword_signals}

  ## Voting Results from Similar Documents
  {state.vote_counts}
  Confidence: {state.confidence:.2%}
  Preliminary category: {state.predicted_category}

  ## Document Content (first {n} pages)
  {state.extracted_text[:4000]}

  ## Top Retrieved Similar Chunks
  {top_3_chunks_with_category}

  ## Task
  Based on all evidence, provide final classification.

  Respond ONLY with this JSON:
  {
    "final_category": "<category>",
    "final_confidence": <0.0-1.0>,
    "reasoning": "<2-3 sentence explanation>",
    "evidence_chunks": ["<key phrase 1>", "<key phrase 2>"]
  }

Parse JSON response and store in state.
```

---

## FastAPI Endpoints

### POST `/ingest`

```
Request:  multipart/form-data
  - file: UploadFile (PDF)
  - category: str (ground truth label — one of CATEGORIES)
  - nct_id: str (optional, NCT number if known)

Process:
  1. Save uploaded file to temp dir
  2. Run Docling extraction (full document, all pages)
  3. Chunk → embed → ingest into Weaviate with metadata
  4. Add to BM25 index

Response:
  {
    "status": "success",
    "filename": "...",
    "category": "Protocol",
    "chunks_ingested": 12,
    "nct_id": "NCT02107703"
  }
```

### POST `/classify`

```
Request:  multipart/form-data
  - file: UploadFile (PDF)

Process:
  1. Save uploaded file to temp dir
  2. Run LangGraph agent (full pipeline)
  3. Return classification result

Response:
  {
    "filename": "unknown_doc.pdf",
    "final_category": "SAP",
    "final_confidence": 0.91,
    "reasoning": "Document contains statistical methodology sections...",
    "fallback_triggered": false,
    "vote_breakdown": {"SAP": 8, "Protocol": 2},
    "keyword_signals": {"SAP": ["statistical analysis plan", "ANCOVA"]},
    "processing_time_seconds": 4.2
  }
```

### POST `/feedback`

```
Request:  application/json
  {
    "filename": "unknown_doc.pdf",
    "predicted_category": "Protocol",
    "correct_category": "SAP",
    "corrected_by": "user@example.com"
  }

Process:
  1. Find chunks in Weaviate by filename
  2. Update category metadata to correct_category
  3. Update source_type to "feedback"
  4. Re-add to BM25 index with correct label
  5. Log correction

Response:
  {
    "status": "updated",
    "chunks_updated": 8,
    "correct_category": "SAP"
  }
```

### GET `/health`

```
Response:
  {
    "status": "ok",
    "weaviate": "connected",
    "chunks_in_db": 1247,
    "categories": {"Protocol": 423, "SAP": 312, "ICF": 287, ...}
  }
```

---

## Bootstrap Ingestion Script

Implement `scripts/bootstrap_ingest.py`:

```
Arguments:
  --input_dir   Path to folder with downloaded PDFs (default: ../clinical_trial_pdfs)
  --batch_size  Number of files per batch (default: 10)

Logic:
  1. Scan input_dir subfolders: 01_Protocol/, 02_SAP/, 03_ICF/, 04_Combined/
  2. Infer category from folder name
  3. For each PDF, call the /ingest endpoint
  4. Print progress: "Ingesting [Protocol] NCT02107703_Prot_000.pdf..."
  5. On error: log and continue (do not crash)
  6. Final summary: total ingested, failed, per-category counts

Run with:
  python scripts/bootstrap_ingest.py --input_dir ../clinical_trial_pdfs
```

---

## Docker Compose

Create `docker-compose.yml` to run Weaviate locally:

```yaml
version: '3.8'
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: ''
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:
```

---

## Requirements

Create `requirements.txt` with these exact packages:

```
# Core
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.9
python-dotenv==1.0.1
pydantic==2.8.2
pydantic-settings==2.4.0

# LangGraph + LLM
langgraph==0.2.28
langchain-anthropic==0.2.4
langchain-core==0.3.8
anthropic==0.34.2

# PDF Extraction
docling==2.5.0

# Embeddings + Reranking
sentence-transformers==3.1.1
torch==2.4.1
transformers==4.44.2

# Vector DB
weaviate-client==4.7.1

# Sparse Search
rank-bm25==0.2.2

# Tokenization
tiktoken==0.7.0

# Utilities
numpy==1.26.4
tqdm==4.66.5
httpx==0.27.2
aiofiles==24.1.0
```

---

## Implementation Order

Build files in EXACTLY this order. Complete each file fully before moving on:

```
Phase 1 — Foundation
  1.  .env.example
  2.  requirements.txt
  3.  docker-compose.yml
  4.  app/__init__.py
  5.  app/core/config.py
  6.  app/core/keywords.py

Phase 2 — Core Processing
  7.  app/core/extractor.py       (Docling wrapper)
  8.  app/core/chunker.py         (page-aware chunking)
  9.  app/core/embedder.py        (SentenceTransformer wrapper)
  10. app/core/reranker.py        (cross-encoder wrapper)

Phase 3 — Database Layer
  11. app/db/weaviate_client.py   (schema + CRUD)
  12. app/db/bm25_store.py        (BM25 in-memory index)

Phase 4 — Agent
  13. app/agent/state.py
  14. app/agent/nodes/extractor.py
  15. app/agent/nodes/retriever.py
  16. app/agent/nodes/classifier.py
  17. app/agent/nodes/deep_reader.py
  18. app/agent/nodes/finalizer.py
  19. app/agent/tools.py
  20. app/agent/graph.py

Phase 5 — API
  21. app/schemas/ingest.py
  22. app/schemas/classify.py
  23. app/schemas/feedback.py
  24. app/api/routes/ingest.py
  25. app/api/routes/classify.py
  26. app/api/routes/feedback.py
  27. app/main.py

Phase 6 — Scripts + Tests
  28. scripts/bootstrap_ingest.py
  29. scripts/test_classify.py
  30. tests/test_chunker.py
  31. tests/test_classifier.py
  32. tests/test_api.py

Phase 7 — Documentation
  33. README.md
```

---

## README Sections to Include

```
1. Project Overview
2. Architecture Diagram (ASCII)
3. Prerequisites (Docker, Python 3.10+)
4. Setup Instructions
   - Clone repo
   - Create .env from .env.example
   - pip install -r requirements.txt
   - docker-compose up -d
   - python scripts/bootstrap_ingest.py
   - uvicorn app.main:app --reload
5. API Usage (curl examples for each endpoint)
6. How Classification Works (plain English)
7. Adding New Document Categories
8. Feedback & Correction Loop
```

---

## Error Handling Rules

- All agent nodes must catch exceptions and set `state.error` — never crash the graph
- FastAPI endpoints return structured error responses with HTTP status codes
- Weaviate connection failure → return 503 with `"weaviate": "unavailable"`
- PDF extraction failure → return 422 with `"error": "Could not extract text from PDF"`
- Low confidence with no fallback improvement → return result with `"warning": "low_confidence"`
- All errors must be logged with filename + traceback

---

## Testing Requirements

- `test_chunker.py` → verify head/tail chunking logic with a sample PDF
- `test_embedder.py` → verify embedding shape and cosine similarity output
- `test_classifier.py` → mock retrieved_chunks, verify voting + confidence math
- `test_api.py` → use FastAPI TestClient, test /ingest → /classify → /feedback flow

---

## Key Constraints

- Do NOT use OpenAI. Only Anthropic Claude Haiku for LLM calls.
- Do NOT use LangChain's built-in vectorstores — use Weaviate client directly.
- Do NOT store raw PDFs in Weaviate — store only text chunks + metadata.
- BM25 index lives in memory — rebuild from Weaviate on app startup.
- All file uploads are temporary — delete from disk after processing.
- Confidence threshold is configurable via .env, default 0.85.
- LLM is called ONLY in the finalizer node — all other nodes are pure Python.