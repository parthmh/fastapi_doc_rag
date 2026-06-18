# FastAPI RAG Documentation

Welcome to the technical documentation for the **FastAPI RAG (Retrieval-Augmented Generation) Service**. 

This application implements a configuration-driven RAG pipeline capable of retrieving relevant sections from FastAPI's official markdown documentation, resolving dynamic code blocks, and synthesizing highly accurate, grounded answers using generative LLMs.

---

## Key Capabilities

*   **Configuration-Driven Ingestion:** Seamlessly switch between light-embedding (MiniLM) and heavy-embedding (Granite) pipelines.
*   **Multimodal Image Ingestion:** A dedicated asynchronous ingestion endpoint (`POST /api/v1/ingest/image`) powered by **FashionCLIP** (`patrickjohncyh/fashion-clip`), utilizing isolated background worker processes, PIL-based in-memory Base64 decoding, and CPU core pinning.
*   **Hybrid Vector Search:** Combines dense vector distance metrics with BM25 sparse keyword queries, resolved using Reciprocal Rank Fusion (RRF).
*   **Dynamic Code Injection:** Automatically resolves and injects filesystem code block placeholders (`{* filepath *}`) during retrieval.
*   **Resilient synthesis:** Integrated OpenAI-compatible client wrapper that handles rate-limiting (`429`) and transient server failures (`5xx`) using exponential backoff and jitter.
*   **Premium Diagnostics Dashboard:** Interactive dark-themed SPA playground displaying search modes, latency metrics, token consumption meters, references, and collapsible reasoning traces.

---

## Quickstart Guide

### 1. Prerequisite Infrastructure
Ensure you have a running instance of Qdrant (by default port `6333`):
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### 2. Install Dependencies
This project uses **`uv`** for dependency management. Sync your environment:
```bash
uv sync
```

### 3. Setup Environment variables
Create a `.env` file at the root of the project:
```ini
RAG_MODEL_TIER=granite                    # Options: minilm, granite (Default: granite)
LLM_PROVIDER=openai                       # Options: openai, gemini (Default: openai)
LLM_MODEL=mistral-small-2506              # Model name (Default: mistral-small-2506)
LLM_BASE_URL=https://api.mistral.ai/v1     # Base URL (Default: https://api.mistral.ai/v1)
OPENAI_API_KEY=your-api-key-here
QDRANT_URL=http://localhost:6333
```

### 4. Initialize & Ingest Database
Run the ingestion script locally to compute embeddings and populate Qdrant (using `--workers` or `-w` to enable parallel threads):
```bash
PYTHONPATH=. .venv/bin/python ingestion/ingest.py --workers 4 --tier minilm
```

### 5. Running the Servers
*   **FastAPI Backend Server:**
    ```bash
    PYTHONPATH=. .venv/bin/uvicorn app.main:app --reload
    ```
*   **Frontend Dashboard Server:**
    ```bash
    python3 -m http.server --directory frontend 8080
    ```

Once started, navigate to **`http://localhost:8080`** to test your queries.

---

## Accessing Interactive & Project Docs

### FastAPI Auto-Generated API Docs
When the FastAPI backend is running, the interactive OpenAPI specifications are automatically served at:
*   **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
*   **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### MkDocs Documentation (This Site)
To run or build this technical documentation site locally using `uv`:
*   **Serve locally**:
    ```bash
    uv run mkdocs serve -a 127.0.0.1:8001
    ```
*   **Build static files**:
    ```bash
    uv run mkdocs build
    ```

