# System Architecture

The FastAPI RAG service organizes components into three primary tiers: the Vector Storage Tier, the API/Business Logic Tier, and the Frontend Visualization Tier.

```mermaid
graph TD
    subgraph Client Tier
        UI[Frontend Client UI]
    end

    subgraph API Tier [FastAPI Backend - Cores 4-7]
        API[FastAPI Main Server]
        QueueText[text_queue: asyncio.Queue]
        QueueImage[image_queue: asyncio.Queue]
    end

    subgraph Background Workers Tier
        SubText[Text Worker Subprocess - Cores 8-11]
        SubImage[Image Worker Subprocess - Cores 12-15]
    end

    subgraph Storage & Upstream Tier
        DB[(Qdrant Vector DB - Cores 0-3)]
        FS[(Local Filesystem /docs_src)]
        LLM[LLM Provider: Gemini / Mistral]
    end

    UI -->|POST /chat| API
    UI -->|POST /api/v1/ingest| API
    UI -->|POST /api/v1/ingest/image| API
    
    API -->|1a. Enqueue Text| QueueText
    API -->|1b. Enqueue Image| QueueImage
    
    QueueText -->|2a. Pipe IPC| SubText
    QueueImage -->|2b. Pipe IPC| SubImage
    
    SubText -->|3a. Bulk Upsert| DB
    SubImage -->|3b. Bulk Upsert| DB
    
    API -->|Search requests| DB
    DB -->|Retrieved candidates| API
    API -->|Resolve code placeholders| FS
    FS -->|Injected code| API
    API -->|Synthesize prompt| LLM
    LLM -->|Text & Token usage| API
    API -->|JSON payload| UI
```

---

## 1. Pipeline Configuration Tiers

The application supports dynamic switching of embedding, chunking, and storage configurations via the `RAG_MODEL_TIER` settings variable.

```mermaid
graph LR
    subgraph MiniLM Tier [MiniLM Configuration]
        M_Doc[Markdown Docs] -->|Chonkie Tokenizer| M_Chunk[220 Token Chunks]
        M_Chunk -->|MiniLM L6 Embeddings| M_Dense[384-dim Dense Vectors]
        M_Dense -->|Indexed| M_Col[(fastapi_doc_rag_minilm)]
    end

    subgraph Granite Tier [Granite Configuration]
        G_Doc[Markdown Docs] -->|Bypassed| G_Node[Full Document Node]
        G_Node -->|Granite Embedding R2| G_Dense[768-dim Dense Vectors]
        G_Dense -->|Indexed| G_Col[(fastapi_doc_rag_granite)]
    end

    subgraph FashionCLIP Tier [FashionCLIP Configuration]
        F_Img[Images / Base64] -->|PIL Image Decode| F_Proc[CLIP Processor]
        F_Proc -->|CLIP CPU Forward Pass| F_Dense[512-dim Dense Vectors]
        F_Dense -->|Indexed| F_Col[(fashion_images_fashion_clip)]
    end
```

### MiniLM Configuration (`minilm`)
*   **Chunking:** Enabled. Documents are chunked into small nodes of at most **220 tokens** using a tokenizer-based splitter (`chonkie`).
*   **Embeddings:** Generates **384-dimensional** dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.
*   **Storage Space:** Writes to and queries from the `fastapi_doc_rag_minilm` Qdrant collection.

### Granite Configuration (`granite`)
*   **Chunking:** Bypassed. Every documentation source file is treated as a single, undivided node.
*   **Embeddings:** Generates **768-dimensional** dense vectors using `ibm-granite/granite-embedding-english-r2`.
*   **Storage Space:** Writes to and queries from the `fastapi_doc_rag_granite` Qdrant collection.

### FashionCLIP Configuration (`fashion_clip`)
*   **Processing:** Subprocess downloads remote URLs concurrently via `ThreadPoolExecutor` or decodes in-memory Base64 strings.
*   **Embeddings:** Generates **512-dimensional** dense vectors using `patrickjohncyh/fashion-clip`.
*   **Storage Space:** Writes to the `fashion_images_fashion_clip` Qdrant collection.

---

## 2. Ingestion Infrastructure
The files within the `ingestion/` directory are executed during the database setup phase or run continuously in the background:
1.  **Parse Markdown / Images:** Reads raw documentation `.md` files or processes incoming API payloads.
2.  **Generate Chunks / Decode Base64:** Chunks text documents (based on model tier) or decodes/normalizes image pixels.
3.  **Calculate Metadata:** Computes file path mappings, section page URLs, product IDs, captions, and token counts.
4.  **Insert Vectors:** Generates embeddings (MiniLM, Granite, or FashionCLIP) and uploads payloads to Qdrant.

