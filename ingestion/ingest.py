

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .chunk_markdown import build_chunks_from_page_tree
from app.config import settings
from .embed_core import embed_and_upsert, ensure_collection
from .parse_markdown import PageTree, build_page

CORPUS_ROOT = Path("corpus")
TUTORIAL_ROOT = CORPUS_ROOT / "tutorial"


@dataclass(slots=True)
class PipelineContext:
    client: QdrantClient
    dense_model: SentenceTransformer
    sparse_model: SparseTextEmbedding
    colbert_model: LateInteractionTextEmbedding


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


def build_page_chunks(page_tree: PageTree) -> list[dict[str, object]]:
    return build_chunks_from_page_tree(page_tree)


def make_context() -> PipelineContext:
    print(f"Loading dense model {settings.dense_model_name}...")
    
    dense_model = SentenceTransformer(settings.dense_model_name, device="cpu")

    print(f"Loading sparse model {settings.sparse_model_name}...")
    
    sparse_model = SparseTextEmbedding(model_name=settings.sparse_model_name)

    print(f"Loading ColBERT model {settings.colbert_model_name}...")
    
    colbert_model = LateInteractionTextEmbedding(
        model_name=settings.colbert_model_name 
    )

    print("All models loaded.")
    return PipelineContext(
        client=QdrantClient(url=settings.qdrant_url),
        dense_model=dense_model,
        sparse_model=sparse_model,
        colbert_model=colbert_model,
    )


def ingest_page(
    markdown_path: Path,
    context: PipelineContext,
) -> PageIngestResult:
    page_start = time.perf_counter()

    page_tree = load_page_tree(markdown_path)
    chunks = build_page_chunks(page_tree)

    if chunks:
        embed_and_upsert(
            chunks,
            client=context.client,
            dense_model=context.dense_model,
            sparse_model=context.sparse_model,
            colbert_model=context.colbert_model,
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
    parser = argparse.ArgumentParser(description="Ingest markdown documentation into Qdrant.")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel worker threads.")
    parser.add_argument("--tier", "-t", type=str, choices=["minilm", "granite"], help="Model configuration tier override.")
    args = parser.parse_args()

    if args.tier:
        settings.rag_model_tier = args.tier

    if not TUTORIAL_ROOT.exists():
        raise FileNotFoundError(f"Tutorial root not found: {TUTORIAL_ROOT}")

    markdown_files = list(iter_markdown_files(TUTORIAL_ROOT))
    if not markdown_files:
        print(f"No markdown files found under {TUTORIAL_ROOT}")
        return

    run_start = time.perf_counter()
    context = make_context()

    print("=" * 100)
    print(f"INGEST START | collection={settings.collection_name} | qdrant={settings.qdrant_url} | workers={args.workers}")
    print(f"DENSE MODEL   : {settings.dense_model_name}")
    print(f"SPARSE MODEL  : {settings.sparse_model_name}")
    print(f"COLBERT MODEL : {settings.colbert_model_name}")
    print(f"TUTORIAL ROOT : {TUTORIAL_ROOT}")
    print(f"PAGES FOUND   : {len(markdown_files)}")
    print("=" * 100)

    # Pre-create collection in main thread to avoid concurrent thread creation race conditions
    print(f"Ensuring Qdrant collection '{settings.collection_name}' is initialized...")
    ensure_collection(
        client=context.client,
        collection_name=settings.collection_name,
        dense_size=settings.dense_vector_size,
        colbert_size=128,  # colbertv2.0 generates 128-dimensional multivectors
    )

    results: list[PageIngestResult] = []
    total_chunks = 0

    if args.workers > 1:
        print(f"Running parallel ingestion with {args.workers} worker threads...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {
                executor.submit(ingest_page, path, context): path
                for path in markdown_files
            }
            for future in as_completed(future_to_file):
                markdown_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    total_chunks += result.chunks
                    print(f"TIME  {markdown_path.relative_to(CORPUS_ROOT)} | {result.seconds:.4f}s (worker)")
                except Exception as e:
                    print(f"ERROR {markdown_path.relative_to(CORPUS_ROOT)} | failed: {e}")
    else:
        print("Running sequential ingestion...")
        for markdown_path in markdown_files:
            try:
                result = ingest_page(markdown_path, context)
                results.append(result)
                total_chunks += result.chunks
                print(f"TIME  {markdown_path.relative_to(CORPUS_ROOT)} | {result.seconds:.4f}s")
            except Exception as e:
                print(f"ERROR {markdown_path.relative_to(CORPUS_ROOT)} | failed: {e}")

    total_elapsed = time.perf_counter() - run_start
    print()
    print("=" * 100)
    print(f"INGEST COMPLETE | pages={len(results)} | chunks={total_chunks}")
    print(f"TOTAL TIME: {total_elapsed:.4f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()