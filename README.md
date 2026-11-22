# Research Paper Assistant - Backend

Backend API service for the **Research Paper Assistant** capstone project, built using **FastAPI**, **Python**, **PyMuPDF**, **Sentence Transformers**, **ChromaDB**, and the **Groq API**.

> **Note**: This repository contains the backend service only. The frontend client will be hosted in a separate repository.

---

## Project Structure

```text
research-paper-assistant-backend/
├── app/
│   ├── __init__.py
│   ├── agents/               # AI Agents for academic intelligence
│   │   ├── __init__.py
│   │   ├── comparator.py     # Cross-paper comparative analysis agent
│   │   ├── gap_analyzer.py   # Explicit limitation & inferred research gap discovery agent
│   │   ├── orchestrator.py   # Central structured LLM caller & JSON parser
│   │   └── summarizer.py     # Direct & hierarchical map-reduce summarizer agent
│   ├── api/                  # API routers
│   │   ├── __init__.py
│   │   ├── chat.py           # POST /chat, POST /chat/{id}, GET /chat/{id}/history, DELETE /chat/{id}
│   │   ├── documents.py      # POST /documents/upload, GET /documents, DELETE /documents/{id}
│   │   ├── health.py         # GET /health
│   │   ├── retrieval.py      # POST /retrieval/search, GET /retrieval/stats
│   │   └── routes/           # Phase 5 research intelligence routes
│   │       ├── __init__.py
│   │       ├── comparison.py # POST /compare
│   │       ├── research_gaps.py # POST /research-gaps
│   │       └── summary.py    # POST /summarize
│   ├── core/                 # Core configurations and utilities
│   │   ├── __init__.py
│   │   ├── config.py         # Application settings loaded from .env
│   │   └── errors.py         # Standardized global JSON error handlers
│   ├── db/                   # Persistence layer
│   │   ├── __init__.py
│   │   └── database.py       # SQLite connection, schema, and document/chunk operations
│   ├── retrieval/            # Vector store & search
│   │   ├── __init__.py
│   │   ├── embeddings.py     # Thread-safe Sentence Transformers embedding service
│   │   ├── indexer.py        # Connects chunking to ChromaDB
│   │   ├── retriever.py      # Semantic chunk retrieval with metadata & cosine distance
│   │   └── vector_store.py   # Persistent ChromaDB client and collection operations
│   ├── schemas/              # Pydantic request and response schemas
│   │   ├── __init__.py
│   │   ├── chat.py           # ChatRequest, ChatResponse, SourceCitation schemas
│   │   ├── comparison.py     # ComparisonRequest, ComparisonResponse schemas
│   │   ├── research_gaps.py  # GapAnalysisRequest, GapAnalysisResponse schemas
│   │   ├── retrieval.py      # Retrieval schemas
│   │   └── summary.py        # SummaryRequest, SummaryResponse schemas
│   └── services/             # Core business logic & AI orchestration
│       ├── __init__.py
│       ├── citation_service.py # Source citation extraction & deduplication
│       ├── comparison_service.py # Cross-paper comparison service
│       ├── context_builder.py  # Structured academic prompt & context formatting
│       ├── gap_analysis_service.py # Explicit & inferred gap analysis service
│       ├── llm_service.py      # Official Groq SDK client wrapper with error handling
│       ├── pdf_loader.py       # Safe PDF extraction via PyMuPDF
│       ├── rag_service.py      # End-to-end RAG question answering orchestrator
│       ├── session_store.py    # In-memory multi-turn conversation manager
│       ├── summary_service.py  # Paper summarization service
│       ├── text_cleaner.py     # Text normalization and paragraph preserving
│       └── text_chunker.py     # Multi-page aware chunking (800 chars, 150 overlap)
├── storage/                  # Persistent data directory (auto-created, git-ignored)
│   ├── uploads/              # Uploaded PDF files
│   ├── chroma/               # ChromaDB persistent vector database
│   └── metadata.db           # SQLite database for document metadata and chunks
├── tests/                    # Automated test suite (56 unit & integration tests)
│   ├── __init__.py
│   ├── conftest.py           # Pytest fixtures and mock generators
│   ├── test_chat.py          # 14 tests for Groq LLM, RAG orchestration, citations, history
│   ├── test_documents_api.py # 7 tests for upload, size limits, list, delete
│   ├── test_pdf_services.py  # 10 tests for loader, cleaner, chunker
│   ├── test_phase5.py        # 10 tests for summarization, comparison, and research gaps
│   └── test_retrieval.py     # 15 tests for embeddings and ChromaDB retrieval
├── .env                      # Local environment configuration (ignored by git)
├── .env.example              # Configuration template
├── .gitignore                # Git ignore rules for Python projects
├── main.py                   # FastAPI entry point with lifespan DB initialization
├── README.md                 # Complete documentation
└── requirements.txt          # Project dependencies
```

---

## Installation & Setup

### 1. Clone or Open the Repository

```bash
cd "research paper assistant (backend)"
```

### 2. Virtual Environment Setup

On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS / Linux:
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration Settings (`.env`)

```env
# Application
APP_NAME="Research Paper Assistant API"
APP_ENV="development"
DEBUG=True
HOST="127.0.0.1"
PORT=8000

# Document Storage & Processing
STORAGE_DIR="./storage"
MAX_UPLOAD_SIZE_MB=20
CHUNK_SIZE=800
CHUNK_OVERLAP=150

# ChromaDB & Embeddings Configuration
CHROMA_PERSIST_DIR="./storage/chroma"
CHROMA_COLLECTION_NAME="research_papers"
EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
DEFAULT_TOP_K=5

# Groq LLM Configuration
GROQ_API_KEY="your-groq-api-key-here"
GROQ_MODEL="llama-3.3-70b-versatile"
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=1024
```

---

## Advanced Research Intelligence Features (Phase 5)

### 1. Research Paper Summarization (`POST /summarize`)
- **Formats Supported**:
  - `short`: 2-3 sentence executive abstract.
  - `detailed`: Comprehensive 2-3 paragraph academic synthesis.
  - `academic`: In-depth methodology and evidence evaluation.
  - `key_points`: Bullet-style breakdown of core contributions.
- **Section Extraction**: Objective, problem statement, methodology, dataset, results, conclusion, limitations, future work.
- **Hierarchical Map-Reduce**: Automatically divides long papers (>10 chunks / 8000 characters) into chunk batches, extracts intermediate representations, and synthesizes the unified final summary.
- **Example Request**:
```json
{
  "document_id": "c7a8b415-3882-45e0-9e6e-c534440ba396",
  "summary_type": "detailed"
}
```

### 2. Research Paper Comparison (`POST /compare`)
- **Input**: 2 to 5 document IDs.
- **Analysis Dimensions**: Research objectives, problem statements, methodologies, datasets, algorithms/models, evaluation metrics, main findings, limitations, future work.
- **Comparative Synthesis**: Surfaces commonalities, contrasts, methodological trade-offs, and an overall comparative overview.
- **Example Request**:
```json
{
  "document_ids": [
    "c7a8b415-3882-45e0-9e6e-c534440ba396",
    "8f8b3df9-c2c3-4d4b-bd22-83951fcfbe9b"
  ]
}
```

### 3. Research Gap Analysis (`POST /research-gaps`)
- **Input**: 1 to 5 document IDs.
- **Rigorous Academic Categorization**:
  - **Explicit Limitations**: Limitations directly acknowledged by authors in the text.
  - **Author-Suggested Future Work**: Specific extensions recommended by original researchers.
  - **Inferred Potential Gaps**: Research opportunities inferred by cross-analyzing multiple papers.
- **Discipline**: Never presents inferred opportunities as global universal facts; explicitly grounds every gap in cited quotations.
- **Example Request**:
```json
{
  "document_ids": [
    "c7a8b415-3882-45e0-9e6e-c534440ba396"
  ]
}
```

---

## Running the Automated Tests

Run the full pytest suite (56 tests across all 5 phases):

```bash
pytest -v
```

All 56 tests execute in ~4 seconds with mock Groq and embedding clients, consuming zero API credits during automated testing.
