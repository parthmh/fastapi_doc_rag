# Concurrent Ingestion Architecture & Implementation

To enable high-speed ingestion that does not pollute our production search index, we implemented a dedicated concurrent ingestion pipeline. This document serves as the detailed implementation reference, profiling data, and benchmark run.

---

## 1. Separate Ingestion Vector Space

To isolate bulk ingestion experiments, load testing, and synthetic data injection from our live production documentation search, we define a separate Qdrant collection space:

*   **Production Collection**: `fastapi_doc_rag_{tier}`
*   **Ingestion Collection**: `fastapi_doc_ingest_minilm` (always uses `multi-qa-MiniLM-L6-cos-v1` for optimal CPU ingestion speed)

During startup, the backend automatically initializes the `fastapi_doc_ingest_minilm` collection with only the 384-dimensional dense vector configuration.

---

## 2. Ingestion Flow and Logic

The ingestion pipeline processes incoming items asynchronously using an in-memory queue to maximize throughput and isolate API response times:

```mermaid
graph TD
    Client[HTTP Client / Load Test] -->|POST /api/v1/ingest| API[FastAPI Ingest Endpoint]
    API -->|Push items| Queue[asyncio.Queue max_size=1.2M]
    API -->|Instant Response 202| Client
    
    subgraph Background Worker
        Queue -->|Instant drain queue.get_nowait| Worker[In-Memory Ingest Worker]
        Worker -->|Batch Vectorization| Embedding[multi-qa-MiniLM-L6-cos-v1 Dense Model]
        Embedding -->|Dense Vectors + Payloads| Qdrant[Qdrant Client bulk upsert]
    end
```

### Steps:
1.  **FastAPI Route (`POST /api/v1/ingest`)**: Receives batch payloads. It validates schemas, generates a task tracking UUID, pushes items into the queue, and returns HTTP status `202 Accepted` immediately (bypassing synchronous wait times).
2.  **Async Queue**: A thread-safe `asyncio.Queue` with a capacity of **1,200,000** elements caches incoming items.
3.  **Background Worker**: Pulls items from the queue. It drains the queue instantly using `queue.get_nowait()` up to batches of size 64 to eliminate event loop context-switching and timer overhead.
4.  **Embedding & Qdrant Upsert**: Runs the CPU-bound dense embedding model in parallel using `anyio.to_thread.run_sync` to avoid blocking the event loop. Constructs Qdrant `PointStruct` objects containing only the dense vector, and executes a batch upsert to `fastapi_doc_ingest_minilm` with `wait=False`.

---

## 3. Pipeline Timing Performance Analysis

We profile the execution time of each stage of the ingestion pipeline per batch of 64 items to identify bottlenecks:

*   **API Enqueuing Latency**: **< 0.5ms** (immediate response to client).
*   **Embedding Generation**: **~50ms - 90ms** (accounting for **~75%** of worker execution time).
*   **Payload Preparation**: **~1ms** (accounting for **~1%** of worker execution time).
*   **Qdrant gRPC Upsert (Network I/O)**: **~15ms - 25ms** (accounting for **~24%** of worker execution time).
*   **Total Worker Latency**: **~70ms - 110ms** per batch of 64 items (indexing throughput of **~570 - 664 points/sec**).

---

## 4. Synthetic Data Generation & Benchmark Tools

To load-test the API under realistic workloads, we developed two custom scripts under `tests/`:

1.  **Zero-Dependency Generator (`tests/generate_synthetic_data.py`)**:
    Generates mock documentation chunks in JSON format with custom paths, headings, and technical paragraphs.
2.  **Benchmark CLI (`tests/benchmark_million_points.py`)**:
    Streams up to 1,000,000 points concurrently in configurable batches and connections.

### Benchmark Run Results (20,000 points):
*   **API Throughput**: **114,672.88 points/second** enqueued.
*   **Total API Acceptance Time**: **0.1744 seconds**.
*   **Average Route Latency**: **15.77 ms** per batch of 200 items.
*   **Peak Indexing Throughput**: **664 points/second**.
*   **Sustained Indexing Throughput**: **570 points/second**.
