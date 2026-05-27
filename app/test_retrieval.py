from __future__ import annotations

from qdrant_client import QdrantClient

from retriever_first_steps import (
    build_retriever,
    RetrievedChunk,
)


QUERIES = [
    "What is the instance of the class FastAPI?",
    "how to create a path operation",
    "how do endpoints work",
    "what is openapi schema",
    "what does decorator info mean",
    "how to declare a get endpoint",
    "how to return json",
    "how to use async endpoint",
    "what are path parameters",
    "endpoint function",
    "api operation",
    "route handler",
]


SEPARATOR = "=" * 140
SUB_SEPARATOR = "-" * 140


def compact_preview(text: str, limit: int = 140) -> str:
    text = text.replace("\n", " ").strip()

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def print_hit(
    label: str,
    rank: int,
    hit: RetrievedChunk,
) -> None:
    payload = hit.payload

    heading = payload.get("heading_text") or payload.get("title") or ""
    chunk_kind = payload.get("chunk_kind")
    node_kind = payload.get("node_kind")
    level = payload.get("level")
    token_count = payload.get("token_count")

    preview = compact_preview(
        payload.get("chunk_text", "")
    )

    raw_score = (
        f"{hit.raw_score:.4f}"
        if hit.raw_score is not None
        else "-"
    )

    boost = (
        f"{hit.boost:.4f}"
        if hit.boost is not None
        else "-"
    )

    print(
        f"{label:<10}"
        f"#{rank:<2} "
        f"score={hit.score:.4f} "
        f"raw={raw_score} "
        f"boost={boost}"
    )

    print(
        f"{'':<10}"
        f"kind={chunk_kind:<12} "
        f"node={node_kind:<12} "
        f"level={str(level):<3} "
        f"tokens={token_count:<4}"
    )

    print(
        f"{'':<10}"
        f"heading={heading}"
    )

    print(
        f"{'':<10}"
        f"{preview}"
    )

    print()


def main() -> None:
    client = QdrantClient(
        url="http://localhost:6333"
    )

    retriever = build_retriever(client)

    for query in QUERIES:
        print()
        print(SEPARATOR)
        print("QUERY:", query)
        print(SEPARATOR)
        print()

        result = retriever.compare(
            query=query,
            limit=5,
            debug=False,
        )

        baseline_ids = [
            hit.chunk_id
            for hit in result.baseline
        ]

        weighted_ids = [
            hit.chunk_id
            for hit in result.weighted
        ]

        print("BASELINE vs WEIGHTED")
        print(SUB_SEPARATOR)
        print()

        for idx in range(5):
            baseline_hit = (
                result.baseline[idx]
                if idx < len(result.baseline)
                else None
            )

            weighted_hit = (
                result.weighted[idx]
                if idx < len(result.weighted)
                else None
            )

            if baseline_hit:
                print_hit(
                    "BASELINE",
                    idx + 1,
                    baseline_hit,
                )

            if weighted_hit:
                print_hit(
                    "WEIGHTED",
                    idx + 1,
                    weighted_hit,
                )

            print(SUB_SEPARATOR)

        moved_in = [
            cid
            for cid in weighted_ids
            if cid not in baseline_ids
        ]

        moved_out = [
            cid
            for cid in baseline_ids
            if cid not in weighted_ids
        ]

        print()
        print("RANKING CHANGES")
        print(SUB_SEPARATOR)

        print("NEW IN WEIGHTED:")
        for cid in moved_in:
            print("  ", cid)

        print()

        print("REMOVED FROM WEIGHTED:")
        for cid in moved_out:
            print("  ", cid)

        print()
        print(SEPARATOR)
        print()


if __name__ == "__main__":
    main()