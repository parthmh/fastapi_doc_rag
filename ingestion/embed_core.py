from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "fastapi_doc_rag"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384
BATCH_SIZE = 32
QDRANT_URL = "http://localhost:6333"


def stable_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )

    for field_name in ("page_id", "node_kind", "chunk_kind"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def make_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    payload = dict(chunk)
    payload.setdefault("chunk_text", chunk.get("chunk_text", ""))
    return payload


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
) -> list[list[float]]:
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def upsert_chunks(
    client: QdrantClient,
    model: SentenceTransformer,
    chunks: list[dict[str, Any]],
) -> None:
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [str(chunk.get("chunk_text", "")) for chunk in batch]
        vectors = embed_texts(model, texts)

        points = [
            models.PointStruct(
                id=stable_point_id(str(chunk["chunk_id"])),
                vector=vector,
                payload=make_payload(chunk),
            )
            for chunk, vector in zip(batch, vectors, strict=True)
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )


def embed_and_upsert(
    chunks: list[dict[str, Any]],
    *,
    client: QdrantClient | None = None,
    model: SentenceTransformer | None = None,
) -> QdrantClient:
    if not chunks:
        return client or QdrantClient(url=QDRANT_URL)

    local_client = client or QdrantClient(url=QDRANT_URL)
    ensure_collection(local_client)

    local_model = model or SentenceTransformer(MODEL_NAME, device="cpu")
    upsert_chunks(local_client, local_model, chunks)

    return local_client