from __future__ import annotations

"""Retrieval helpers for the FastAPI docs RAG pipeline.

This module provides two retrieval strategies for side-by-side comparison:

1. Baseline dense retrieval
2. Weighted retrieval using:
   - hierarchy signals from existing chunk metadata
   - token-count percentile signals computed from heading chunks only

We intentionally do NOT require any ingestion-format changes yet.
The current chunk JSON already contains the fields we need:
- chunk_kind
- node_kind
- level
- heading_text
- kind
- title
- token_count
- parent_id
- page_id

Context reconstruction from first_steps_tree.json remains a later step.
"""

from dataclasses import dataclass
from pathlib import Path
import bisect
import json
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "fastapi_doc_rag"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LIMIT = 5
PREFETCH_LIMIT = 25
CHUNKS_PATH = Path("processed") / "first_steps_chunks.json"
TREE_PATH = Path("processed") / "first_steps_tree.json"

# Modest hierarchy weights.
HEADING_LEVEL_WEIGHTS: dict[int, float] = {
    1: 0.18,
    2: 0.15,
    3: 0.10,
    4: 0.06,
    5: 0.03,
}

# Admonitions should contribute very little.
ADMONITION_BOOST_WITH_TITLE = 0.02
ADMONITION_BOOST_NO_TITLE = 0.01

# Percentile bands for heading token-count weighting only.
# This is intentionally small so semantic similarity remains primary.
HEADING_PERCENTILE_BAND_SCORES: list[tuple[float, float]] = [
    (0.10, -0.04),
    (0.30, 0.00),
    (0.70, 0.05),
    (0.90, 0.02),
    (1.01, -0.03),
]


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    payload: dict[str, Any]
    raw_score: float | None = None
    boost: float | None = None
    hierarchy_boost: float | None = None
    length_boost: float | None = None
    length_percentile: float | None = None


@dataclass(slots=True)
class RetrievalResult:
    query: str
    baseline: list[RetrievedChunk]
    weighted: list[RetrievedChunk]


class HeadingPercentileStats:
    """Compute token-count percentiles from heading chunks only."""

    def __init__(self, chunks_path: Path = CHUNKS_PATH) -> None:
        self.chunks_path = chunks_path
        self._heading_counts: list[int] = []
        self._load()

    def _load(self) -> None:
        if not self.chunks_path.exists():
            return

        with self.chunks_path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk in chunks:
            if str(chunk.get("chunk_kind") or chunk.get("node_kind") or "") != "heading":
                continue
            token_count = int(chunk.get("token_count") or 0)
            self._heading_counts.append(token_count)

        self._heading_counts.sort()

    def percentile(self, token_count: int) -> float:
        if not self._heading_counts:
            return 0.5

        idx = bisect.bisect_right(self._heading_counts, token_count)
        return idx / len(self._heading_counts)

    @staticmethod
    def band_score(percentile: float) -> float:
        for cutoff, score in HEADING_PERCENTILE_BAND_SCORES:
            if percentile <= cutoff:
                return score
        return 0.0


class Retriever:
    def __init__(
        self,
        client: QdrantClient,
        model: SentenceTransformer | None = None,
        collection_name: str = COLLECTION_NAME,
        chunks_path: Path = CHUNKS_PATH,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.model = model or SentenceTransformer(MODEL_NAME, device="cpu")
        self.stats = HeadingPercentileStats(chunks_path=chunks_path)

    def encode_query(self, query: str) -> list[float]:
        vector = self.model.encode(query, normalize_embeddings=True)
        return vector.tolist()

    def _as_hit(self, point: Any) -> RetrievedChunk:
        payload = dict(point.payload or {})
        return RetrievedChunk(
            chunk_id=str(payload.get("chunk_id", point.id)),
            score=float(point.score),
            payload=payload,
        )

    def _hierarchy_boost(self, payload: dict[str, Any]) -> float:
        chunk_kind = str(payload.get("chunk_kind") or payload.get("node_kind") or "")
        node_kind = str(payload.get("node_kind") or "")
        level = int(payload.get("level") or 0)
        title = payload.get("title")

        if chunk_kind == "admonition" or node_kind == "admonition":
            return ADMONITION_BOOST_WITH_TITLE if title else ADMONITION_BOOST_NO_TITLE

        if chunk_kind == "heading" or node_kind == "heading":
            return HEADING_LEVEL_WEIGHTS.get(level, 0.0)

        return 0.0

    def _length_boost(self, payload: dict[str, Any]) -> tuple[float, float]:
        chunk_kind = str(payload.get("chunk_kind") or payload.get("node_kind") or "")
        if chunk_kind != "heading":
            return 0.0, 0.5

        token_count = int(payload.get("token_count") or 0)
        percentile = self.stats.percentile(token_count)
        score = self.stats.band_score(percentile)
        return score, percentile

    def _score_hit(self, hit: RetrievedChunk) -> RetrievedChunk:
        hierarchy_boost = self._hierarchy_boost(hit.payload)
        length_boost, percentile = self._length_boost(hit.payload)
        adjusted = hit.score + hierarchy_boost + length_boost

        return RetrievedChunk(
            chunk_id=hit.chunk_id,
            score=adjusted,
            payload=hit.payload,
            raw_score=hit.score,
            boost=hierarchy_boost + length_boost,
            hierarchy_boost=hierarchy_boost,
            length_boost=length_boost,
            length_percentile=percentile,
        )

    def retrieve_baseline(self, query: str, limit: int = DEFAULT_LIMIT) -> list[RetrievedChunk]:
        query_vector = self.encode_query(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return [self._as_hit(point) for point in results.points]

    def retrieve_weighted(self, query: str, limit: int = DEFAULT_LIMIT, debug: bool = False) -> list[RetrievedChunk]:
        query_vector = self.encode_query(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=max(limit, PREFETCH_LIMIT),
            with_payload=True,
        )

        rescored = [self._score_hit(self._as_hit(point)) for point in results.points]
        rescored.sort(key=lambda item: item.score, reverse=True)

        if debug:
            print()
            print("RAW CANDIDATES")
            print("-" * 100)
            for item in rescored:
                print(
                    f"raw={item.raw_score:.4f} "
                    f"hier={item.hierarchy_boost:.4f} "
                    f"len={item.length_boost:.4f} "
                    f"pct={item.length_percentile:.2f} "
                    f"adjusted={item.score:.4f} "
                    f"kind={item.payload.get('chunk_kind')} "
                    f"node_kind={item.payload.get('node_kind')} "
                    f"level={item.payload.get('level')} "
                    f"title={item.payload.get('title')} "
                    f"id={item.chunk_id}"
                )

            print()
            print("RESCORED TOP")
            print("-" * 100)
            for item in rescored[:limit]:
                print(
                    f"adjusted={item.score:.4f} raw={item.raw_score:.4f} "
                    f"hier={item.hierarchy_boost:.4f} len={item.length_boost:.4f} "
                    f"pct={item.length_percentile:.2f} "
                    f"kind={item.payload.get('chunk_kind')} "
                    f"node_kind={item.payload.get('node_kind')} "
                    f"level={item.payload.get('level')} "
                    f"title={item.payload.get('title')} "
                    f"id={item.chunk_id}"
                )

        return rescored[:limit]

    def compare(self, query: str, limit: int = DEFAULT_LIMIT, debug: bool = False) -> RetrievalResult:
        baseline = self.retrieve_baseline(query, limit=limit)
        weighted = self.retrieve_weighted(query, limit=limit, debug=debug)
        return RetrievalResult(query=query, baseline=baseline, weighted=weighted)

    def print_comparison(self, query: str, limit: int = DEFAULT_LIMIT, debug: bool = False) -> None:
        result = self.compare(query, limit=limit, debug=debug)

        print()
        print("=" * 100)
        print(f"QUERY: {result.query}")
        print("=" * 100)

        print()
        print("BASELINE")
        print("-" * 100)
        for idx, hit in enumerate(result.baseline, start=1):
            self._print_hit(idx, hit)

        print()
        print("WEIGHTED")
        print("-" * 100)
        for idx, hit in enumerate(result.weighted, start=1):
            self._print_hit(idx, hit)

        print()
        print("DELTA NOTES")
        print("-" * 100)
        baseline_ids = [hit.chunk_id for hit in result.baseline]
        weighted_ids = [hit.chunk_id for hit in result.weighted]
        moved_in = [cid for cid in weighted_ids if cid not in baseline_ids]
        moved_out = [cid for cid in baseline_ids if cid not in weighted_ids]
        print(f"New in weighted: {moved_in}")
        print(f"Dropped by weighting: {moved_out}")

    def _print_hit(self, rank: int, hit: RetrievedChunk) -> None:
        payload = hit.payload
        heading = payload.get("heading_text") or payload.get("title") or ""
        node_kind = payload.get("node_kind", "")
        chunk_kind = payload.get("chunk_kind", "")
        level = payload.get("level", "")
        token_count = payload.get("token_count", "")
        pct = f"{hit.length_percentile:.2f}" if hit.length_percentile is not None else "-"
        preview = str(payload.get("chunk_text", ""))[:220].replace("\n", " ")

        print(f"{rank:>2}. score={hit.score:.4f} | chunk_id={hit.chunk_id}")
        print(
            f"    chunk_kind={chunk_kind} node_kind={node_kind} level={level} "
            f"pct={pct} tokens={token_count} heading={heading!r}"
        )
        print(f"    {preview}")

    def load_tree(self, tree_path: Path = TREE_PATH) -> dict[str, Any]:
        with tree_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def build_context_chain(self, chunk_payload: dict[str, Any], tree: dict[str, Any]) -> list[dict[str, Any]]:
        _ = chunk_payload
        _ = tree
        return []


def build_retriever(client: QdrantClient) -> Retriever:
    return Retriever(client=client)


def compare_query(
    client: QdrantClient,
    query: str,
    limit: int = DEFAULT_LIMIT,
    debug: bool = False,
) -> RetrievalResult:
    retriever = build_retriever(client)
    return retriever.compare(query, limit=limit, debug=debug)
