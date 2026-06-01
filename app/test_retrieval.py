from __future__ import annotations

from app.retriever import Retriever


QUERIES = [
    "give me an example of Dependency Injection",
    "how to implement CORS?",
    "what are Multiple Models in FastAPI?",
    "how to use pydantic's .model_dump() method",
    "give me an example of application structure",
]


def short(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_result_block(
    title: str,
    results,
) -> None:
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
            f"kind={payload.get('chunk_kind')} | "
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


def compare_results(
    baseline,
    weighted,
) -> None:
    print()
    print("-" * 120)
    print("RANK COMPARISON")
    print("-" * 120)

    max_len = max(len(baseline), len(weighted))

    for idx in range(max_len):
        left = baseline[idx] if idx < len(baseline) else None
        right = weighted[idx] if idx < len(weighted) else None

        print()

        print(f"RANK {idx + 1}")

        if left:
            left_payload = left.payload
            print(
                f"BASELINE | "
                f"{left.score:.4f} | "
                f"{left_payload.get('chunk_kind')} | "
                f"{left_payload.get('heading_text')}"
            )

        if right:
            right_payload = right.payload
            print(
                f"WEIGHTED | "
                f"{right.score:.4f} | "
                f"{right_payload.get('chunk_kind')} | "
                f"{right_payload.get('heading_text')}"
            )


def main() -> None:
    retriever = Retriever()

    for query in QUERIES:
        print()
        print()
        print("#" * 140)
        print("QUERY:", query)
        print("#" * 140)

        baseline = retriever.search(query=query, limit=5)
        weighted = retriever.search_weighted(query=query, limit=5)

        print_result_block(
            "BASELINE RESULTS",
            baseline,
        )

        print_result_block(
            "WEIGHTED RESULTS",
            weighted,
        )

        compare_results(
            baseline,
            weighted,
        )


if __name__ == "__main__":
    main()