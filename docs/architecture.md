# System Architecture

The FastAPI RAG service organizes components into three primary tiers: the Vector Storage Tier, the API/Business Logic Tier, and the Frontend Visualization Tier.

```mermaid
graph TD
    UI[Frontend Client UI] -->|HTTP POST /chat| API[FastAPI Backend]
    API -->|Search requests| DB[(Qdrant Vector DB)]
    DB -->|Retrieved candidates| API
    API -->|Resolve code placeholders| FS[(Local Filesystem /docs_src)]
    FS -->|Injected code| API
    API -->|Synthesize prompt| LLM[LLM Provider: Gemini / Mistral]
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
```

### MiniLM Configuration (`minilm`)
*   **Chunking:** Enabled. Documents are chunked into small nodes of at most **220 tokens** using a tokenizer-based splitter (`chonkie`).
*   **Embeddings:** Generates **384-dimensional** dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.
*   **Storage Space:** Writes to and queries from the `fastapi_doc_rag_minilm` Qdrant collection.

### Granite Configuration (`granite`)
*   **Chunking:** Bypassed. Every documentation source file is treated as a single, undivided node.
*   **Embeddings:** Generates **768-dimensional** dense vectors using `ibm-granite/granite-embedding-english-r2`.
*   **Storage Space:** Writes to and queries from the `fastapi_doc_rag_granite` Qdrant collection.

---

## 2. Ingestion Infrastructure
The files within the `ingestion/` directory are executed during the database setup phase:
1.  **Parse Markdown:** Reads raw documentation `.md` files.
2.  **Generate Chunks:** Based on the active model configuration tier, either chunks the documents or keeps them whole.
3.  **Calculate Metadata:** Computes absolute file path mappings, page URLs, headers, and token counts.
4.  **Insert Vectors:** Generates embeddings for chunks and uploads payloads to Qdrant.
