from __future__ import annotations

import asyncio
import uuid
import time
from typing import Any
from anyio.to_thread import run_sync
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.retriever import Retriever
from app.schemas import IngestItem
from ingestion.embed_core import (
    embed_dense_texts,
    embed_sparse_texts,
    embed_colbert_texts,
    ensure_collection,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    COLBERT_VECTOR_NAME,
    stable_point_id,
)

# Shared singleton for the MiniLM model
_minilm_model: SentenceTransformer | None = None

def get_minilm_model() -> SentenceTransformer:
    """
    Retrieves or loads the sentence-transformers/multi-qa-MiniLM-L6-cos-v1 model.
    Enforces MiniLM only for ingestion to keep indexing fast.
    """
    global _minilm_model
    if _minilm_model is None:
        print("Loading MiniLM model ('sentence-transformers/multi-qa-MiniLM-L6-cos-v1') on CPU for ingestion worker...")
        _minilm_model = SentenceTransformer(
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            device="cpu"
        )
    return _minilm_model

def ensure_ingest_collection_initialized(retriever: Retriever) -> None:
    """
    Checks if the isolated ingestion collection exists in Qdrant; if not, creates it.
    Uses ONLY dense vector configuration with 384 dimensions for MiniLM.
    """
    collection_name = settings.ingest_collection_name
    if retriever.client.collection_exists(collection_name):
        return

    retriever.client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            )
        }
    )
    for field_name in ("page_id", "node_kind", "chunk_kind"):
        retriever.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

def process_ingest_batch(batch: list[IngestItem], retriever: Retriever) -> None:
    """
    Synchronous function that encodes a batch of chunks using MiniLM and upserts them to Qdrant.
    Measures latency for embedding vs. request upserts.
    """
    if not batch:
        return

    texts = [item.chunk_text for item in batch]
    minilm_model = get_minilm_model()

    # --- Phase 1: Embedding Generation ---
    start_embed = time.perf_counter()
    
    dense_vectors = embed_dense_texts(minilm_model, texts)
    
    embed_latency = time.perf_counter() - start_embed

    # --- Phase 2: Point Construction & Metadata Extraction ---
    start_prep = time.perf_counter()
    collection_name = settings.ingest_collection_name
    points = []
    
    for item, dense in zip(
        batch, dense_vectors, strict=True
    ):
        # 1. Token Metric Extraction (Calculated using MiniLM model tokenizer)
        token_count = item.token_count
        if token_count is None:
            if hasattr(minilm_model, "tokenizer") and minilm_model.tokenizer is not None:
                token_count = len(minilm_model.tokenizer.encode(item.chunk_text, add_special_tokens=False))
            else:
                token_count = len(item.chunk_text.split())

        # 2. Determine chunk ID
        chunk_id = item.chunk_id
        if not chunk_id:
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{item.page_id}/{item.heading_text}/{item.chunk_text[:30]}"))

        payload = {
            "chunk_text": item.chunk_text,
            "heading_text": item.heading_text,
            "page_id": item.page_id,
            "section_url": item.section_url,
            "chunk_id": chunk_id,
            "node_kind": item.node_kind or "section",
            "chunk_kind": item.chunk_kind or "content",
            "token_count": token_count,
        }

        points.append(
            models.PointStruct(
                id=stable_point_id(chunk_id),
                vector={
                    DENSE_VECTOR_NAME: dense,
                },
                payload=payload,
            )
        )
    prep_latency = time.perf_counter() - start_prep

    # --- Phase 3: Qdrant Batch Upsert ---
    start_qdrant = time.perf_counter()
    
    retriever.client.upsert(
        collection_name=collection_name,
        points=points,
        wait=False,
    )
    
    qdrant_latency = time.perf_counter() - start_qdrant
    
    # Log detailed timing statistics for performance isolation in the exact user-specified format
    model_ms = embed_latency * 1000
    io_ms = qdrant_latency * 1000
    print(f"Model  : {model_ms:.0f}ms")
    print(f"io task: {io_ms:.0f}ms")

async def ingest_worker_loop(
    queue: asyncio.Queue,
    retriever: Retriever,
    batch_size: int = 64,
    timeout_sec: float = 0.1,
) -> None:
    """
    Continuous background loop consuming document items from the queue,
    batching them, and executing parallel ingestion.
    """
    print("Background Ingestion Worker initialized.")
    # Warm up MiniLM model is now handled synchronously in the main thread during lifespan startup.

    while True:
        try:
            item = await queue.get()
            batch = [item]

            # Drain the queue up to batch_size as quickly as possible without yielding
            while len(batch) < batch_size:
                try:
                    batch.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            try:
                await run_sync(process_ingest_batch, batch, retriever)
            except Exception as e:
                print(f"Error processing async ingestion batch: {e}")
            finally:
                for _ in range(len(batch)):
                    queue.task_done()

        except asyncio.CancelledError:
            print("Background Ingestion Worker cancelled.")
            break
        except Exception as e:
            print(f"Unhandled exception in ingest worker: {e}")
            await asyncio.sleep(1.0)
