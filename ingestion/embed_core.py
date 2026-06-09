from __future__ import annotations

import uuid
from typing import Any

from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


from app.config import settings

COLLECTION_NAME = settings.collection_name

DENSE_MODEL_NAME = settings.dense_model_name
SPARSE_MODEL_NAME = settings.sparse_model_name
COLBERT_MODEL_NAME = settings.colbert_model_name

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
COLBERT_VECTOR_NAME = "colbert"

BATCH_SIZE = 32
QDRANT_URL = settings.qdrant_url


def stable_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def make_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    payload = dict(chunk)
    payload.setdefault("chunk_text", chunk.get("chunk_text", ""))
    return payload


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    *,
    dense_size: int,
    colbert_size: int,
) -> None:
    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=dense_size,
                distance=models.Distance.COSINE,
            ),
            COLBERT_VECTOR_NAME: models.VectorParams(
                size=colbert_size,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )

    for field_name in ("page_id", "node_kind", "chunk_kind"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def _to_dense_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(x) for x in vector]


def _to_multivector(vector: Any) -> list[list[float]]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [[float(x) for x in row] for row in vector]


def _to_sparse_vector(vector: Any) -> models.SparseVector:
    if isinstance(vector, models.SparseVector):
        return vector

    if hasattr(vector, "indices") and hasattr(vector, "values"):
        indices_raw = getattr(vector, "indices", None)
        values_raw = getattr(vector, "values", None)
        if indices_raw is None or values_raw is None:
            raise TypeError("Sparse embedding is missing indices or values")

        return models.SparseVector(
            indices=[int(i) for i in list(indices_raw)],
            values=[float(v) for v in list(values_raw)],
        )

    if isinstance(vector, dict):
        indices_raw = vector.get("indices")
        if indices_raw is None:
            indices_raw = vector.get("index")
        values_raw = vector.get("values")
        if values_raw is None:
            values_raw = vector.get("value")
        if indices_raw is None or values_raw is None:
            raise TypeError("Sparse embedding is missing indices or values")

        return models.SparseVector(
            indices=[int(i) for i in list(indices_raw)],
            values=[float(v) for v in list(values_raw)],
        )

    raise TypeError(f"Unsupported sparse embedding type: {type(vector)!r}")


def embed_dense_texts(
    model: SentenceTransformer,
    texts: list[str],
) -> list[list[float]]:
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [_to_dense_list(vector) for vector in vectors]


def embed_sparse_texts(
    model: SparseTextEmbedding,
    texts: list[str],
) -> list[models.SparseVector]:
    embeddings = list(model.embed(texts, batch_size=BATCH_SIZE))
    return [_to_sparse_vector(embedding) for embedding in embeddings]


def embed_colbert_texts(
    model: LateInteractionTextEmbedding,
    texts: list[str],
) -> list[list[list[float]]]:
    embeddings = list(model.passage_embed(texts))
    return [_to_multivector(embedding) for embedding in embeddings]


def upsert_chunks(
    client: QdrantClient,
    dense_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    colbert_model: LateInteractionTextEmbedding,
    chunks: list[dict[str, Any]],
) -> None:
    if not chunks:
        return

    collection_name = settings.collection_name
    collection_ready = client.collection_exists(collection_name)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [str(chunk.get("chunk_text", "")) for chunk in batch]

        dense_vectors = embed_dense_texts(dense_model, texts)
        sparse_vectors = embed_sparse_texts(sparse_model, texts)
        colbert_vectors = embed_colbert_texts(colbert_model, texts)

        if not collection_ready:
            ensure_collection(
                client,
                collection_name=collection_name,
                dense_size=len(dense_vectors[0]),
                colbert_size=len(colbert_vectors[0][0]),
            )
            collection_ready = True

        points = []
        for chunk, dense_vector, sparse_vector, colbert_vector in zip(
            batch,
            dense_vectors,
            sparse_vectors,
            colbert_vectors,
            strict=True,
        ):
            points.append(
                models.PointStruct(
                    id=stable_point_id(str(chunk["chunk_id"])),
                    vector={
                        DENSE_VECTOR_NAME: dense_vector,
                        SPARSE_VECTOR_NAME: sparse_vector,
                        COLBERT_VECTOR_NAME: colbert_vector,
                    },
                    payload=make_payload(chunk),
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
        )


def embed_and_upsert(
    chunks: list[dict[str, Any]],
    *,
    client: QdrantClient | None = None,
    dense_model: SentenceTransformer | None = None,
    sparse_model: SparseTextEmbedding | None = None,
    colbert_model: LateInteractionTextEmbedding | None = None,
) -> QdrantClient:
    if not chunks:
        return client or QdrantClient(url=QDRANT_URL)

    local_client = client or QdrantClient(url=QDRANT_URL)
    local_dense_model = dense_model or SentenceTransformer(settings.dense_model_name, device="cpu")
    local_sparse_model = sparse_model or SparseTextEmbedding(settings.sparse_model_name)
    local_colbert_model = colbert_model or LateInteractionTextEmbedding(settings.colbert_model_name)

    upsert_chunks(
        local_client,
        local_dense_model,
        local_sparse_model,
        local_colbert_model,
        chunks,
    )

    return local_client