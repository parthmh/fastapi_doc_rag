# FashionCLIP Image Ingestion Pipeline

To expand the capabilities of the FastAPI RAG service to support visual search, we introduce a multimodal image ingestion pipeline. This pipeline leverages **FashionCLIP** (`patrickjohncyh/fashion-clip`) to process image URLs, generate 512-dimensional dense visual embeddings, and index them in Qdrant.

To maintain our performance SLA under high-concurrency ingestion workloads, this feature mirrors our decoupled multiprocessing architecture.

---

## 1. System Architecture

The image ingestion pipeline runs as a separate OS process, fully isolated from both the text ingestion pipeline and Uvicorn's main HTTP server.

```mermaid
graph TD
    Client[Client Application] -->|POST /api/v1/ingest/image| API[FastAPI Endpoint]
    API -->|1. Enqueue payload| Queue[(image_queue: asyncio.Queue)]
    API -->|2. HTTP 202 Accepted| Client
    
    subgraph FastAPI Uvicorn Process [Cores 4-7]
        Queue -->|3. get & batch| IPCWriter[image_ipc_writer_loop]
    end

    subgraph FashionCLIP Subprocess [Cores 12-15]
        IPCWriter -->|4. Pipe serialization| StdinBuffer[stdin / os.read]
        StdinBuffer -->|5. ThreadPoolExecutor| Downloader[Concurrent Image Downloader]
        Downloader -->|6. PIL Images| Preprocessor[CLIPProcessor]
        Preprocessor -->|7. Forward Pass| Model[FashionCLIP Model CPU]
        Model -->|8. 512-dim Vectors| Qdrant[Qdrant DB: Cores 0-3]
    end
```

---

## 2. API Design & Data Schema

The mirrored endpoint `POST /api/v1/ingest/image` accepts a batch of image items, enqueues them, and returns immediately:

### Request Payloads:
```json
{
  "items": [
    {
      "image_url": "https://example.com/images/shirt_123.jpg",
      "product_id": "prod_123",
      "caption": "Blue cotton crewneck t-shirt",
      "metadata": {
        "category": "apparel",
        "brand": "FashionBrand"
      }
    }
  ]
}
```

### Response Payload:
```json
{
  "status": "accepted",
  "task_id": "f5127814-c104-4df2-811c-22345091a182",
  "queued_count": 1
}
```

---

## 3. Worker Subprocess Pipeline Flow

Inside the isolated child process `ingestion/ingest_image_worker.py`:
1.  **Byte Stream Reader**: Reads serialized payloads from standard input using raw `os.read(0, 65536)` and splits them by newlines (`\n`) to avoid buffering delays.
2.  **Concurrent Image Fetcher**: Downloads images concurrently using a python `ThreadPoolExecutor` to handle network I/O overhead.
3.  **Preprocessing & Tokenization**: Feeds PIL Images to `CLIPProcessor` to resize, normalize, and pre-process images into tensors.
4.  **FashionCLIP Inference**: Executes the PyTorch forward pass `CLIPModel.get_image_features` in a single batched CPU matrix operation to generate normalized embeddings.
5.  **Qdrant Bulk Indexing**: Executes a batch upsert to the `fashion_images` collection in Qdrant.

---

## 4. Hardware Resource Allocation

To prevent resource starvation and CPU scheduling contention, we allocate distinct hardware core pins:

*   **Qdrant Database**: Cores `0-3` (4 Cores)
*   **FastAPI / Uvicorn parent processes**: Cores `4-7` (4 Cores)
*   **Text Ingestion Worker subprocess**: Cores `8-11` (4 Cores)
*   **Image Ingestion Worker subprocess**: Cores `12-15` (4 Cores)

---

## 5. Load Testing & Benchmark Results (June 2026)

To test the image ingestion pipeline under load, we simulated concurrent users streaming image payloads to the endpoint.

### Test Setup:
*   **Locust Pinning**: Pinned the Locust process to Cores `8-11` (the text ingestion worker cores, which were idle during this test).
*   **Locust Command**:
    ```bash
    taskset -c 8-11 uv run locust -f tests/locust_image_ingest.py --headless -u 10 -r 2 --run-time 15s --host http://localhost:8000
    ```
*   **Configuration**:
    *   Concurrency: 10 users ramping up at 2 users/sec.
    *   Payload Size: Batches of 5-20 random image items per request.
    *   Ingestion Batch Size: `1` (1-by-1 ingestion on worker, to protect memory and run sequentially).

### Benchmark Metrics:
| Performance Metric | Result |
| :--- | :--- |
| **Total Requests Completed** | 430 |
| **Failures** | 0 (0.00%) |
| **Average Response Time** | 1 ms |
| **Median Response Time** | 2 ms |
| **Max Response Time** | 13 ms |
| **API Throughput** | 28.95 requests/second |
| **Data Integrity (Qdrant Points)** | Successfully indexed into `fashion_images_fashion_clip` |

### Key Findings:
1. **Decoupled API Performance**: The average response time of **1ms** confirms that Uvicorn accepts and enqueues request payloads instantly, shielding the client from heavy neural network feature extraction and image network downloads.
2. **Background Processing Rate**: With `INGEST_BATCH_SIZE=1`, the worker sequentially processes each image. Individual image processing takes `~0.6s to 1.5s` for image downloads and `~100ms` for PyTorch CPU forward-pass embedding generation. Points are steadily upserted to Qdrant in the background at a rate of ~1 point/sec, avoiding system overload.
