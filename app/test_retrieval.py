from __future__ import annotations

from app.retriever import Retriever


QUERIES = [
    "give me an example of Dependency Injection",
    "how to implement CORS?",
    "what are Multiple Models in FastAPI?",
    "how to use pydantic's .model_dump() method",
    "give me an example of application structure",
    "how to convert a data type to JSON?",
    "how to convert a data type to json?",
    "How to create a middleware?",
    "What happens when a client tries to access a non-existent resource?",
    "How to override the httpexception error handling?",
]


def short(text: str, limit: int = 1000) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_result_block(title: str, results) -> None:
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)

    for idx, item in enumerate(results, start=1):
        payload = item.payload

        print()
        print(
            f"[{idx}] "
            f"score={item.score:.4f} | "
            f"chunk_kind={payload.get('chunk_kind')} | "
            f"node_kind={payload.get('node_kind')} | "
            f"admonition_kind={payload.get('kind')}"
        )

        print(f"page_id      : {payload.get('page_id')}")
        print(f"heading      : {payload.get('heading_text')}")
        print(f"section_url  : {payload.get('section_url')}")
        print(f"chunk_id     : {payload.get('chunk_id')}")

        print()
        print(short(str(payload.get("chunk_text", ""))))
        print()


def compare_rankings(dense, sparse, hybrid, hybrid_rerank) -> None:
    print()
    print("-" * 140)
    print("RANK COMPARISON")
    print("-" * 140)

    max_len = max(
        len(dense),
        len(sparse),
        len(hybrid),
        len(hybrid_rerank),
    )

    for idx in range(max_len):
        print()
        print(f"RANK {idx + 1}")

        if idx < len(dense):
            payload = dense[idx].payload
            print(
                f"DENSE         | "
                f"{dense[idx].score:.4f} | "
                f"{payload.get('heading_text')}"
            )

        if idx < len(sparse):
            payload = sparse[idx].payload
            print(
                f"SPARSE        | "
                f"{sparse[idx].score:.4f} | "
                f"{payload.get('heading_text')}"
            )

        if idx < len(hybrid):
            payload = hybrid[idx].payload
            print(
                f"HYBRID        | "
                f"{hybrid[idx].score:.4f} | "
                f"{payload.get('heading_text')}"
            )

        if idx < len(hybrid_rerank):
            payload = hybrid_rerank[idx].payload
            print(
                f"HYBRID+RERANK | "
                f"{hybrid_rerank[idx].score:.4f} | "
                f"{payload.get('heading_text')}"
            )


def main() -> None:
    retriever = Retriever()

    for query in QUERIES:
        print()
        print()
        print("#" * 140)
        print("QUERY:", query)
        print("#" * 140)

        dense_results = retriever.search(
            query=query,
            mode="dense",
            limit=5,
            prefetch_limit=20,
        )

        sparse_results = retriever.search(
            query=query,
            mode="sparse",
            limit=5,
            prefetch_limit=20,
        )

        hybrid_results = retriever.search(
            query=query,
            mode="hybrid",
            limit=5,
            prefetch_limit=20,
        )

        hybrid_rerank_results = retriever.search(
            query=query,
            mode="hybrid_rerank",
            limit=5,
            prefetch_limit=20,
            rerank_limit=10,
        )

        print_result_block("DENSE RESULTS", dense_results)
        print_result_block("SPARSE RESULTS", sparse_results)
        print_result_block("HYBRID RESULTS", hybrid_results)
        print_result_block("HYBRID + COLBERT RERANK RESULTS", hybrid_rerank_results)

        compare_rankings(
            dense_results,
            sparse_results,
            hybrid_results,
            hybrid_rerank_results,
        )


if __name__ == "__main__":
    main()