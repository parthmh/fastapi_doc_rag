# Antigravity Agent Guidelines: FastAPI RAG Refactor

## Project Architecture Context
* **Core Framework**: FastAPI runtime system placed inside the `app/` folder.
* **Storage & Vectors**: Indexed via local `qdrant/` instances and isolated indices.
* **Pipeline Infrastructure**: Chunking routines and documents are stored inside `ingestion/`. Intermediary operational logs go to `processed/`.
* **Environment Controls**: Initialized and updated strictly via `pyproject.toml` and managed by `uv`.

## Target Optimization Objective
Refactor dense embedding and chunking layers to handle configuration-driven toggling:
1. **Baseline**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, Chunking Enabled at 256 tokens) -> Target: `fastapi_doc_rag_minilm`
2. **Challenger**: `ibm-granite/granite-embedding-english-r2` (768 dimensions, Chunking Bypassed) -> Target: `fastapi_doc_rag_granite`

## Chunking and Metadata Constraints
* **Token Metric Extraction**: When chunking is bypassed under the Granite configuration tier, you must still calculate token count properties per node structure. Pass this into the standard schema dictionary for retrieval validation workflows.
* **Vector Schema Isolation**: Dynamically suffix the active model identifier onto all Qdrant collection targets to prevent dimension layout failures.
* **Zero Autonomous Overwrites**: The AI engine is strictly forbidden from executing silent filesystem writes.
* **Human Authorization Gate**: Stop execution and prompt the user for validation approval (`Ctrl + R`) before running any file-modifying tools.

## Project Refactoring State (June 2026)
* **Active Tier Configuration**: Handled dynamically via `pydantic-settings` in [config.py](file:///home/ad.rapidops.com/parth.patel/learn/projects/fastapi_doc_rag/app/config.py).
* **Collection Schema Suffixing**: Switched from static `fastapi_doc_rag` to `fastapi_doc_rag_minilm` and `fastapi_doc_rag_granite`.
* **Ingestion Status**: Both collections are fully populated and tested.
