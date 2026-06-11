from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from app.config import Settings, settings as global_settings
from app.schemas import SearchMode
from ingestion.embed_core import (
    COLBERT_VECTOR_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class Retriever:
    def __init__(
        self,
        client: QdrantClient | None = None,
        dense_model: SentenceTransformer | None = None,
        sparse_model: SparseTextEmbedding | None = None,
        colbert_model: LateInteractionTextEmbedding | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or global_settings
        self.client = client or QdrantClient(url=self.settings.qdrant_url)

        self.dense_model = dense_model or SentenceTransformer(
            self.settings.dense_model_name,
            device="cpu",
        )

        self.sparse_model = sparse_model or SparseTextEmbedding(
            model_name=self.settings.sparse_model_name,
            threads=1,
        )

        self.colbert_model = colbert_model or LateInteractionTextEmbedding(
            model_name=self.settings.colbert_model_name,
            threads=1,
        )

    def encode_dense_query(self, query: str) -> list[float]:
        return self.dense_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

    def encode_sparse_query(self, query: str):
        return list(self.sparse_model.query_embed(query))[0]

    def encode_colbert_query(self, query: str):
        return list(self.colbert_model.query_embed(query))[0]

    def _to_sparse_vector(
        self,
        sparse_embedding: Any,
    ) -> models.SparseVector:
        return models.SparseVector(
            indices=list(sparse_embedding.indices),
            values=list(sparse_embedding.values),
        )

    def _to_results(self, points: Any) -> list[RetrievedChunk]:
        output: list[RetrievedChunk] = []

        for point in points:
            payload = dict(point.payload or {})

            output.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    score=float(point.score),
                    payload=payload,
                )
            )

        return output

    def _admonition_multiplier(self, payload: dict[str, Any]) -> float:
        chunk_kind = str(payload.get("chunk_kind") or "")
        node_kind = str(payload.get("node_kind") or "")
        kind = str(payload.get("kind") or "")

        if (
            chunk_kind == "admonition"
            or node_kind == "admonition"
            or kind == "admonition"
        ):
            return 0.6

        return 1.0

    def _apply_admonition_penalty(
        self,
        results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        adjusted: list[RetrievedChunk] = []

        for item in results:
            adjusted.append(
                RetrievedChunk(
                    chunk_id=item.chunk_id,
                    score=item.score * self._admonition_multiplier(item.payload),
                    payload=item.payload,
                )
            )

        adjusted.sort(key=lambda item: item.score, reverse=True)
        return adjusted

    def _dense_search(
        self,
        query: str,
        limit: int,
        fetch_limit: int,
    ) -> list[RetrievedChunk]:
        query_vector = self.encode_dense_query(query)

        results = self.client.query_points(
            collection_name=self.settings.collection_name,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=fetch_limit,
            with_payload=True,
        )

        return self._apply_admonition_penalty(
            self._to_results(results.points)
        )[:limit]

    def _sparse_search(
        self,
        query: str,
        limit: int,
        fetch_limit: int,
    ) -> list[RetrievedChunk]:
        sparse_query = self.encode_sparse_query(query)

        results = self.client.query_points(
            collection_name=self.settings.collection_name,
            query=self._to_sparse_vector(sparse_query),
            using=SPARSE_VECTOR_NAME,
            limit=fetch_limit,
            with_payload=True,
        )

        return self._apply_admonition_penalty(
            self._to_results(results.points)
        )[:limit]

    def _hybrid_search(
        self,
        query: str,
        limit: int,
        prefetch_limit: int,
    ) -> list[RetrievedChunk]:
        dense_query = self.encode_dense_query(query)
        sparse_query = self.encode_sparse_query(query)

        results = self.client.query_points(
            collection_name=self.settings.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=self._to_sparse_vector(sparse_query),
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            limit=prefetch_limit,
            with_payload=True,
        )

        return self._apply_admonition_penalty(
            self._to_results(results.points)
        )[:limit]

    def _hybrid_rerank_search(
        self,
        query: str,
        limit: int,
        prefetch_limit: int,
        rerank_limit: int,
    ) -> list[RetrievedChunk]:
        dense_query = self.encode_dense_query(query)
        sparse_query = self.encode_sparse_query(query)
        colbert_query = self.encode_colbert_query(query)

        results = self.client.query_points(
            collection_name=self.settings.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=self._to_sparse_vector(sparse_query),
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
            ],
            query=colbert_query,
            using=COLBERT_VECTOR_NAME,
            limit=rerank_limit,
            with_payload=True,
        )

        return self._apply_admonition_penalty(
            self._to_results(results.points)
        )[:limit]

    def search(
        self,
        query: str,
        mode: SearchMode = "dense",
        limit: int = 5,
        prefetch_limit: int = 20,
        rerank_limit: int = 10,
    ) -> list[RetrievedChunk]:
        if mode == "dense":
            return self._dense_search(
                query=query,
                limit=limit,
                fetch_limit=prefetch_limit,
            )

        if mode == "sparse":
            return self._sparse_search(
                query=query,
                limit=limit,
                fetch_limit=prefetch_limit,
            )

        if mode == "hybrid":
            return self._hybrid_search(
                query=query,
                limit=limit,
                prefetch_limit=prefetch_limit,
            )

        if mode == "hybrid_rerank":
            return self._hybrid_rerank_search(
                query=query,
                limit=limit,
                prefetch_limit=prefetch_limit,
                rerank_limit=rerank_limit,
            )

        raise ValueError(f"Unsupported mode: {mode}")