from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from ingestion.embed_core import COLLECTION_NAME, MODEL_NAME, QDRANT_URL


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class Retriever:
    def __init__(
        self,
        client: QdrantClient | None = None,
        model: SentenceTransformer | None = None,
    ) -> None:
        self.client = client or QdrantClient(url=QDRANT_URL)
        self.model = model or SentenceTransformer(MODEL_NAME, device="cpu")

    def encode_query(self, query: str) -> list[float]:
        return self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

    def _to_results(self, points) -> list[RetrievedChunk]:
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

    def search(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        query_vector = self.encode_query(query)

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return self._to_results(results.points)

    def search_weighted(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        query_vector = self.encode_query(query)

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=models.Prefetch(
                query=query_vector,
                limit=100,
            ),
            query=models.FormulaQuery(
                formula=models.SumExpression(
                    sum=[
                        models.MultExpression(
                            mult=[
                                "$score",
                                models.SumExpression(
                                    sum=[
                                        1.0,
                                        models.MultExpression(
                                            mult=[
                                                -0.4,
                                                models.FieldCondition(
                                                    key="node_kind",
                                                    match=models.MatchAny(any=["admonition"]),
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ]
                )
            ),
            limit=limit,
            with_payload=True,
        )

        return self._to_results(results.points)