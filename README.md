# FastAPI RAG Service

A configuration-driven Retrieval-Augmented Generation (RAG) service designed to parse, chunk, index, and query FastAPI's official documentation. It features hybrid search (BM25 + Dense) with Reciprocal Rank Fusion (RRF), late-interaction reranking (ColBERT), dynamic code block resolution, and a beautiful dark-mode web playground.

---

## 🚀 Quickstart

### 1. Setup Local Environment
This project uses **`uv`** for Python package management.

```bash
# Clone the repository and sync dependencies
uv sync
```

### 2. Configure Environment Variables
Create a `.env` file in the root of the project:
```ini
RAG_MODEL_TIER=minilm                     # Options: minilm, granite
LLM_PROVIDER=openai                       # Options: openai, gemini
LLM_MODEL=mistral-small-2506              # Model name
LLM_BASE_URL=https://api.mistral.ai/v1     # Base URL for API requests
OPENAI_API_KEY=your_key                   # API key (for Mistral/OpenAI provider)
GEMINI_API_KEY=your_key                   # API key (for Gemini provider)
QDRANT_URL=http://localhost:6333           # Qdrant DB connection
```

### 3. Initialize/Ingest Database
Run the ingestion script locally to parse the markdown pages, compute embeddings, and populate Qdrant. You can speed it up using parallel workers:
```bash
# Ingest using 4 parallel workers on the minilm tier
PYTHONPATH=. .venv/bin/python ingestion/ingest.py --workers 4 --tier minilm
```
Available arguments:
*   `--workers`, `-w`: Number of parallel thread workers (default: `1`).
*   `--tier`, `-t`: Model configuration tier override (`minilm` or `granite`).


### 4. Run the Servers
*   **FastAPI Backend Server**:
    ```bash
    PYTHONPATH=. .venv/bin/uvicorn app.main:app --reload
    ```
*   **Frontend Client Playground**:
    ```bash
    python3 -m http.server --directory frontend 8080
    ```
    Then visit **[http://localhost:8080](http://localhost:8080)**.

---

## 📖 Accessing Documentation

This project provides comprehensive documentation for API routes, system architecture, dataflows, and load test results.

### A. FastAPI Auto-Generated API Docs
When the FastAPI backend is running locally on port `8000`, you can access interactive API specifications directly in your browser:
*   **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
*   **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### B. MkDocs Project Documentation
The static site documentation is managed by MkDocs. You don't need to install it locally in your virtualenv; you can run it dynamically with `uv`:
*   **Serve Documentation Locally**:
    ```bash
    uv run --with mkdocs --with mkdocs-material mkdocs serve -a 127.0.0.1:8001
    ```
    Open your browser and visit **[http://127.0.0.1:8001](http://127.0.0.1:8001)**.
*   **Build Static Site**:
    ```bash
    uv run --with mkdocs --with mkdocs-material mkdocs build
    ```
    This builds the production-ready static site inside the `site/` folder.

---

## 🐳 Dockerized Deployment & Database Initialization

Standard practice for containerized RAG services is to spin up the database (Qdrant) and application backend together, and then run a one-time database schema creation and data ingestion job.

### 1. Build and Run Cluster
```bash
# Ensure local conflicting database instances are stopped, then build/run services
docker compose up --build -d
```
This starts:
1.  **`qdrant`**: Qdrant DB listening on `http://localhost:6333`.
2.  **`backend`**: FastAPI application on `http://localhost:8000`.
3.  **`frontend`**: Static web server playing dashboard playground on `http://localhost:8080`.

### 2. Populate the Database (One-time Ingestion)
Since the Docker Compose volume is initially empty, you must run the ingestion pipeline inside the running backend container. You can run it concurrently to speed up the process:
```bash
# Run multi-threaded ingestion inside the container (e.g. 4 workers on the granite tier)
docker compose exec backend python -m ingestion.ingest --workers 4 --tier granite
```

---

## 🛠️ Tech Stack & Architecture

*   **FastAPI Backend**: Organized type-safe endpoint schemas, OpenAPI details, and health-checks.
*   **Qdrant Vector Database**: Houses dense vector fields, BM25 sparse indexes, and payload metadata.
*   **Chonkie & Sentence-Transformers**: Handles dynamic text splitting and fast CPU embedding.
*   **ColBERT Cross-Encoder**: Reranks documents using fine-grained token late-interaction.
*   **Locust Load Testing**: Simulates concurrent users querying `/chat` to test pipeline backoff resilience.
