# RAG Pipeline Dataflow

This section describes the dynamic end-to-end execution flow of a query through the retrieval and synthesis modules.

```mermaid
sequenceDiagram
    autonumber
    actor User as Client UI
    participant API as FastAPI Backend
    participant QDR as Qdrant DB
    participant LFS as Local Filesystem
    participant Upstream as Upstream LLM (Mistral/Gemini)

    User->>API: POST /chat (query, strategy, thinking)
    API->>QDR: Vector Search (Dense / Sparse / Hybrid / Rerank)
    QDR-->>API: Top 3 candidate chunks
    API->>LFS: Read corpus/docs_src/ reference files
    API->>API: Replace code placeholders ({* path *}) with raw code
    API->>Upstream: Send prompt (with retry queue enabled)
    Note over Upstream: If 429 rate limit or 5xx occur,<br/>backend retries with exponential backoff
    Upstream-->>API: Synthesized text + token usage
    API-->>User: ChatResponse JSON (text, reference metadata, performance metrics)
```

---

## 1. Vector Retrieval Strategies

The user can select from four retrieval strategies on the frontend:

### A. Dense Search
Queries the dense vector index in Qdrant using Cosine Similarity. The query is converted into embeddings (MiniLM or Granite) and matched against stored node embeddings.

### B. Sparse Search
Executes keyword matching inside Qdrant's sparse index using the BM25 model, ideal for finding exact class, function, or keyword names.

### C. Hybrid Search
Performs both Dense Search and Sparse Search concurrently, merging candidates using **Reciprocal Rank Fusion (RRF)** to combine semantic and keyword-match relevance.

### D. Hybrid + Rerank
Runs Hybrid Search to retrieve candidate chunks, then feeds the top candidates through a local **ColBERT late-interaction cross-encoder model** (`colbert-ir/colbertv2.0`) to rerank the results based on deep token-level query interactions. The top 3 reranked results are returned.

---

## 2. Context Reconstruction & Code Injection

Standard vector chunks often refer to source code files (e.g., `{* tutorial/cors/src/main.py *}`). If fed directly to the LLM, the model would lack the actual code context.

Our backend addresses this via a **Dynamic Code Injection** handler:
1.  **Parse Placeholders:** Identifies files referenced within `{* ... *}` tags in the retrieved chunks.
2.  **Filesystem Lookup:** Resolves the path relative to the `corpus/docs_src/` root directory.
3.  **File Reading:** Reads the raw python code from disk.
4.  **Token Optimization:** Replaces the placeholder with the actual code contents, collapses duplicate newlines (`\n{3,}` $\rightarrow$ `\n\n`) to preserve token boundaries, and compiles the final structured prompt context.

---

## 3. Resilient Upstream Communication

To handle rate limits and transient gateway errors from free-tier providers (such as Mistral or Gemini), the backend implements a resilient HTTP client wrapper:
*   **Monitored Status Codes:** Retries on `429 Too Many Requests` and standard `500`-`504` server errors.
*   **Header Inspection:** Dynamically waits for the length specified in the standard `Retry-After` header if sent by the provider.
*   **Backoff & Jitter:** If the header is absent, waits using exponential backoff with random jitter (`base_delay * 2^attempt + random(0, 0.5)` seconds) for up to 5 attempts before raising a gateway timeout.
