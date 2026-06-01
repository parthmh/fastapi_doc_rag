from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from ingestion.embed_core import (
    DEFAULT_COLLECTION_NAME,
    MODEL_NAME,
    QDRANT_URL,
)


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class Retriever:
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client: QdrantClient | None = None,
        model: SentenceTransformer | None = None,
    ) -> None:
        self.collection_name = collection_name
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
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return self._to_results(results.points)

    def search_weighted(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        query_vector = self.encode_query(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
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

    def print_search(self, query: str, limit: int = 5) -> None:
        results = self.search(query=query, limit=limit)

        print()
        print("=" * 100)
        print(f"QUERY: {query}")
        print(f"COLLECTION: {self.collection_name}")
        print("=" * 100)

        for idx, item in enumerate(results, start=1):
            payload = item.payload
            print()
            print(f"{idx}. score={item.score:.4f}")
            print(f"   chunk_id={item.chunk_id}")
            print(f"   page_id={payload.get('page_id')}")
            print(f"   heading={payload.get('heading_text')}")
            print(f"   tokens={payload.get('token_count')}")
            print(f"   kind={payload.get('kind')}")
            print(f"   title={payload.get('title')}")
            print()
            print(str(payload.get("chunk_text", ""))[:500])