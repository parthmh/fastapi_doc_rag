# Load Testing & Deployment

This section covers how to execute benchmarks using Locust and how to containerize the complete application cluster using Docker.

---

## 1. Load Testing with Locust

We utilize **Locust** to benchmark our API and measure how the RAG pipeline processes concurrent queries.

### Test Configuration (`tests/locustfile.py`)
*   **Behavior:** Simulates users querying `/chat` with random search modes and queries from a test dataset, alongside health checks.
*   **Think Time:** Simulated users wait between **1.0 to 3.0 seconds** between requests.
*   **Concurrency limits:** Designed to benchmark rates up to and exceeding Mistral's 5 requests per second (RPS) threshold.

### Running Locust Tests (Headless)
Run a headless benchmark from your terminal:
```bash
.venv/bin/locust -f tests/locustfile.py --headless -u 20 -r 4 -t 30s --host http://localhost:8000
```
*   `-u 20`: Runs 20 concurrent simulated users (~10 req/s total).
*   `-r 4`: Spawns 4 users per second.
*   `-t 30s`: Runs the load test for 30 seconds.

---

## 2. Containerized Deployment

To deploy the entire RAG pipeline in production, we containerize all services using Docker.

### Dockerfile
The backend uses a multi-stage Docker build to compile virtual environments using `uv`, producing a lightweight final image:

```dockerfile
# Stage 1: Build virtual environment
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Stage 2: Final runtime image
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY app/ /app/app/
COPY corpus/ /app/corpus/
COPY ingestion/ /app/ingestion/
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
Use a Docker Compose configuration to orchestrate the backend, Qdrant database, and static file server hosting the frontend playground:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant_db
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  backend:
    build: .
    container_name: rag_backend
    ports:
      - "8000:8000"
    environment:
      - QDRANT_URL=http://qdrant:6333
      - RAG_MODEL_TIER=granite
      - LLM_PROVIDER=openai
      - LLM_MODEL=mistral-small-2506
      - LLM_BASE_URL=https://api.mistral.ai/v1
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - /tmp/fastembed_cache:/tmp/fastembed_cache
    depends_on:
      - qdrant

  frontend:
    image: halverneus/static-file-server:latest
    container_name: rag_frontend
    ports:
      - "8080:8080"
    volumes:
      - ./frontend:/web
    environment:
      - PORT=8080
      - FOLDER=/web

volumes:
  qdrant_data:
```

Launch the cluster in the background:
```bash
docker compose up -d
```
The RAG backend will start on port `8000`, Qdrant on `6333`, and the playground dashboard will be accessible at **`http://localhost:8080`**.

### 3. Initialize & Ingest Vector Data
Since the Qdrant database service starts empty, you must run the ingestion script inside the running backend container to generate embeddings and populate the collection. 

To run this in the **most efficient way**, utilize parallel worker threads to hide network roundtrip latency to Qdrant:
```bash
# Recommended for MiniLM tier (4 threads)
docker compose exec backend python -m ingestion.ingest --workers 4 --tier minilm

# Recommended for Granite tier (12 threads)
docker compose exec backend python -m ingestion.ingest --workers 12 --tier granite
```

#### Ingestion Efficiency Insights
*   **MiniLM Ingestion**: Speeds up to **~67 seconds** with 4 workers. Since MiniLM inference is fast, the bottleneck is network uploads to Qdrant; 4 threads provide the optimal overlap between file reads, CPU matrix multiplication, and Qdrant network uploads.
*   **Granite Ingestion**: Speeds up to **~291 seconds** with 12 workers. Because Granite is heavily CPU-bound, individual documents take longer due to CPU thread contention (e.g., 29s sequential vs 255s under 12-thread load). However, using 12 threads completely saturates Qdrant's network throughput and maximizes parallel uploads, yielding an overall 11.5% runtime saving.
*   **Lightweight CPU Image**: By configuring PyTorch to compile with CPU-only wheels (removing unused NVIDIA CUDA/Triton binaries), the virtual environment shrank from **5.2GB to 1.4GB** and the final backend Docker image was reduced to just **517MB**.
