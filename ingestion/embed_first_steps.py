from __future__ import annotations

from pathlib import Path
import json
from typing import Any
import uuid

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


PROCESSED_DIR = Path("processed")
CHUNKS_PATH = PROCESSED_DIR / "first_steps_chunks.json"

COLLECTION_NAME = "fastapi_doc_rag"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384
BATCH_SIZE = 32

QDRANT_URL = "http://localhost:6333"


def stable_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def load_chunks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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

        texts = [
            str(chunk.get("chunk_text", ""))
            for chunk in batch
        ]

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


def print_collection_info(client: QdrantClient) -> None:
    info = client.get_collection(COLLECTION_NAME)

    print()
    print("=" * 80)
    print("COLLECTION INFO")
    print("=" * 80)
    print(info)


def run_query(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    limit: int = 5,
) -> None:
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
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
        print(f"heading={payload.get('heading_text')}")
        print(f"tokens={payload.get('token_count')}")
        print()
        print(str(payload.get("chunk_text", ""))[:500])


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)

    if not chunks:
        print(f"No chunks found in {CHUNKS_PATH}")
        return

    client = QdrantClient(url=QDRANT_URL)

    ensure_collection(client)

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    upsert_chunks(client, model, chunks)

    print()
    print(
        f"Embedded and upserted "
        f"{len(chunks)} chunks into "
        f"{COLLECTION_NAME} at {QDRANT_URL}"
    )

    print_collection_info(client)

    queries = [
        "What is the instance of the class FastAPI?",
        "how to create a path operation",
        "how do endpoints work",
        "what is openapi schema",
        "what does decorator info mean",
    ]

    for query in queries:
        run_query(
            client=client,
            model=model,
            query=query,
        )


if __name__ == "__main__":
    main()