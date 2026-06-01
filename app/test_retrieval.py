from __future__ import annotations

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.retriever import Retriever
from ingestion.embed_core import MODEL_NAME, QDRANT_URL


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
    "How to stream SSE?",
    "Can you implement SSE with POST?"
]

BASELINE_COLLECTION = "fastapi_doc_rag_section"
NEW_COLLECTION = "fastapi_doc_rag_h2_subtree"


def short(text: str, limit: int = 1000) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_result_block(
    title: str,
    collection_name: str,
    results,
) -> None:
    print()
    print("=" * 120)
    print(f"{title} | {collection_name}")
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
    new,
    baseline_name: str,
    new_name: str,
) -> None:
    print()
    print("-" * 120)
    print("RANK COMPARISON")
    print("-" * 120)

    max_len = max(len(baseline), len(new))

    for idx in range(max_len):
        left = baseline[idx] if idx < len(baseline) else None
        right = new[idx] if idx < len(new) else None

        print()
        print(f"RANK {idx + 1}")

        if left:
            left_payload = left.payload
            print(
                f"{baseline_name:<12} | "
                f"{left.score:.4f} | "
                f"{left_payload.get('chunk_kind')} | "
                f"{left_payload.get('heading_text')}"
            )

        if right:
            right_payload = right.payload
            print(
                f"{new_name:<12} | "
                f"{right.score:.4f} | "
                f"{right_payload.get('chunk_kind')} | "
                f"{right_payload.get('heading_text')}"
            )


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    baseline_retriever = Retriever(
        collection_name=BASELINE_COLLECTION,
        client=client,
        model=model,
    )
    new_retriever = Retriever(
        collection_name=NEW_COLLECTION,
        client=client,
        model=model,
    )

    for query in QUERIES:
        print()
        print()
        print("#" * 140)
        print("QUERY:", query)
        print("#" * 140)

        baseline = baseline_retriever.search(query=query, limit=5)
        new = new_retriever.search(query=query, limit=5)

        print_result_block(
            "BASELINE RESULTS",
            BASELINE_COLLECTION,
            baseline,
        )

        print_result_block(
            "NEW RESULTS",
            NEW_COLLECTION,
            new,
        )

        compare_results(
            baseline,
            new,
            "BASELINE",
            "NEW",
        )


if __name__ == "__main__":
    main()