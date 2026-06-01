from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .chunk_markdown import build_chunks_from_page_tree
from .embed_core import (
    DEFAULT_COLLECTION_NAME,
    MODEL_NAME,
    QDRANT_URL,
    embed_and_upsert,
)
from .parse_markdown import PageTree, build_page

CORPUS_ROOT = Path("corpus")
TUTORIAL_ROOT = CORPUS_ROOT / "tutorial"


@dataclass(slots=True)
class PipelineContext:
    client: QdrantClient
    model: SentenceTransformer
    collection_name: str
    chunk_strategy: str


@dataclass(slots=True)
class PageIngestResult:
    page_id: str
    source_file: str
    chunks: int
    seconds: float


def iter_markdown_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.md"))


def load_page_tree(markdown_path: Path) -> PageTree:
    markdown_text = markdown_path.read_text(encoding="utf-8")
    return build_page(markdown_text, markdown_path)


def build_page_chunks(page_tree: PageTree, strategy: str) -> list[dict[str, object]]:
    return build_chunks_from_page_tree(page_tree, strategy=strategy)


def make_context(collection_name: str, chunk_strategy: str) -> PipelineContext:
    return PipelineContext(
        client=QdrantClient(url=QDRANT_URL),
        model=SentenceTransformer(MODEL_NAME, device="cpu"),
        collection_name=collection_name,
        chunk_strategy=chunk_strategy,
    )


def ingest_page(
    markdown_path: Path,
    context: PipelineContext,
) -> PageIngestResult:
    page_start = time.perf_counter()

    page_tree = load_page_tree(markdown_path)
    chunks = build_page_chunks(page_tree, strategy=context.chunk_strategy)

    if chunks:
        embed_and_upsert(
            chunks,
            client=context.client,
            model=context.model,
            collection_name=context.collection_name,
        )
        print(
            f"DONE  {page_tree.page.source_file} | "
            f"page_id={page_tree.page.page_id} | chunks={len(chunks)}"
        )
    else:
        print(f"SKIP  {page_tree.page.source_file} | no chunks")

    elapsed = time.perf_counter() - page_start
    return PageIngestResult(
        page_id=page_tree.page.page_id,
        source_file=page_tree.page.source_file,
        chunks=len(chunks),
        seconds=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest FastAPI docs into Qdrant.")
    parser.add_argument(
        "--chunk-strategy",
        choices=["section", "h2_subtree"],
        default="section",
        help="Chunking strategy to use.",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Qdrant collection name. Defaults to fastapi_doc_rag_<chunk-strategy>.",
    )
    args = parser.parse_args()

    collection_name = args.collection_name or f"fastapi_doc_rag_{args.chunk_strategy}"

    if not TUTORIAL_ROOT.exists():
        raise FileNotFoundError(f"Tutorial root not found: {TUTORIAL_ROOT}")

    markdown_files = list(iter_markdown_files(TUTORIAL_ROOT))
    if not markdown_files:
        print(f"No markdown files found under {TUTORIAL_ROOT}")
        return

    run_start = time.perf_counter()
    context = make_context(
        collection_name=collection_name,
        chunk_strategy=args.chunk_strategy,
    )

    print("=" * 100)
    print(f"INGEST START | collection={collection_name} | qdrant={QDRANT_URL}")
    print(f"STRATEGY: {args.chunk_strategy}")
    print(f"TUTORIAL ROOT: {TUTORIAL_ROOT}")
    print(f"PAGES FOUND: {len(markdown_files)}")
    print("=" * 100)

    results: list[PageIngestResult] = []
    total_chunks = 0

    for markdown_path in markdown_files:
        result = ingest_page(markdown_path, context)
        results.append(result)
        total_chunks += result.chunks
        print(f"TIME  {markdown_path.relative_to(CORPUS_ROOT)} | {result.seconds:.4f}s")

    total_elapsed = time.perf_counter() - run_start
    print()
    print("=" * 100)
    print(f"INGEST COMPLETE | pages={len(results)} | chunks={total_chunks}")
    print(f"TOTAL TIME: {total_elapsed:.4f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()