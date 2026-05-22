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


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def upsert_chunks(client: QdrantClient, model: SentenceTransformer, chunks: list[dict[str, Any]]) -> None:
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

        client.upsert(collection_name=COLLECTION_NAME, points=points)


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    if not chunks:
        print(f"No chunks found in {CHUNKS_PATH}")
        return

    client = QdrantClient(":memory:")
    ensure_collection(client)

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    upsert_chunks(client, model, chunks)

    print(f"Embedded and upserted {len(chunks)} chunks into {COLLECTION_NAME} in memory")

    query = "What is the instance of the class FastAPI?"

    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5,
    )

    print()
    print("=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    for point in results.points:
        payload = point.payload

        print()
        print(f"score={point.score:.4f}")
        print(f"chunk_id={payload['chunk_id']}")
        print(f"heading={payload.get('heading_text')}")
        print(f"tokens={payload.get('token_count')}")
        print()
        print(payload["chunk_text"][:500])


if __name__ == "__main__":
    main()
