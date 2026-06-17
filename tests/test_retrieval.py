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


def compare_tiers(query: str, minilm_results: list, granite_results: list) -> None:
    print()
    print("-" * 140)
    print(f"TIER COMPARISON FOR QUERY: {query}")
    print("-" * 140)
    
    max_len = max(len(minilm_results), len(granite_results))
    print(f"{'Rank':<5} | {'MINILM (score | chunk_id | heading)':<65} | {'GRANITE (score | chunk_id | heading)':<65}")
    print("-" * 140)
    
    for idx in range(max_len):
        minilm_str = ""
        granite_str = ""
        
        if idx < len(minilm_results):
            r = minilm_results[idx]
            payload = r.payload
            heading = payload.get("heading_text") or payload.get("title") or ""
            heading_trunc = heading[:30] + "..." if len(heading) > 33 else heading
            minilm_str = f"{r.score:.4f} | {r.chunk_id[-12:]} | {heading_trunc}"
            
        if idx < len(granite_results):
            r = granite_results[idx]
            payload = r.payload
            heading = payload.get("heading_text") or payload.get("title") or ""
            heading_trunc = heading[:30] + "..." if len(heading) > 33 else heading
            granite_str = f"{r.score:.4f} | {r.chunk_id[-12:]} | {heading_trunc}"
            
        print(f"{idx+1:<5} | {minilm_str:<65} | {granite_str:<65}")


def main() -> None:
    from app.config import Settings
    
    # Instantiate retrievers for both tiers
    retriever_minilm = Retriever(settings=Settings(rag_model_tier="minilm"))
    retriever_granite = Retriever(settings=Settings(rag_model_tier="granite"))

    for query in QUERIES:
        print()
        print()
        print("#" * 140)
        print("QUERY:", query)
        print("#" * 140)

        minilm_results = retriever_minilm.search(
            query=query,
            mode="hybrid_rerank",
            limit=5,
            prefetch_limit=20,
            rerank_limit=10,
        )

        granite_results = retriever_granite.search(
            query=query,
            mode="hybrid_rerank",
            limit=5,
            prefetch_limit=20,
            rerank_limit=10,
        )

        compare_tiers(query, minilm_results, granite_results)


if __name__ == "__main__":
    main()