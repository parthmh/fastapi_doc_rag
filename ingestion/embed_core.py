from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


DEFAULT_COLLECTION_NAME = "fastapi_doc_rag"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384
BATCH_SIZE = 32
QDRANT_URL = "http://localhost:6333"


def stable_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def ensure_collection(client: QdrantClient, collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )

    for field_name in ("page_id", "node_kind", "chunk_kind"):
        client.create_payload_index(
            collection_name=collection_name,
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
    collection_name: str = DEFAULT_COLLECTION_NAME,
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
            collection_name=collection_name,
            points=points,
        )


def embed_and_upsert(
    chunks: list[dict[str, Any]],
    *,
    client: QdrantClient | None = None,
    model: SentenceTransformer | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> QdrantClient:
    if not chunks:
        return client or QdrantClient(url=QDRANT_URL)

    local_client = client or QdrantClient(url=QDRANT_URL)
    ensure_collection(local_client, collection_name=collection_name)

    local_model = model or SentenceTransformer(MODEL_NAME, device="cpu")
    upsert_chunks(
        local_client,
        local_model,
        chunks,
        collection_name=collection_name,
    )

    return local_client


def run_query(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    limit: int = 5,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    print()
    print("=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    for point in results.points:
        payload = point.payload or {}

        print()
        print(f"score={point.score:.4f}")
        print(f"chunk_id={payload.get('chunk_id')}")
        print(f"page_id={payload.get('page_id')}")
        print(f"heading={payload.get('heading_text')}")
        print(f"tokens={payload.get('token_count')}")
        print()
        print(str(payload.get("chunk_text", ""))[:500])


if __name__ == "__main__":
    raise SystemExit(
        "This module is a reusable embedding core. Use a runner script to load chunks and call embed_and_upsert()."
    )