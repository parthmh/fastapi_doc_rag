from __future__ import annotations

from pathlib import Path
import time
from typing import Iterable

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from chunk_markdown import build_chunks_from_page_tree
from embed_core import COLLECTION_NAME, MODEL_NAME, QDRANT_URL, embed_and_upsert
from parse_markdown import PageTree, build_page


CORPUS_ROOT = Path("corpus")
TUTORIAL_ROOT = CORPUS_ROOT / "tutorial"


def iter_markdown_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.md"))


def load_page_tree(markdown_path: Path) -> PageTree:
    markdown_text = markdown_path.read_text(encoding="utf-8")
    return build_page(markdown_text, markdown_path)


def build_page_chunks(page_tree: PageTree) -> list[dict[str, object]]:
    return build_chunks_from_page_tree(page_tree)


def ingest_page(
    markdown_path: Path,
    client: QdrantClient,
    model: SentenceTransformer,
) -> int:
    page_tree = load_page_tree(markdown_path)
    chunks = build_page_chunks(page_tree)

    if not chunks:
        print(f"SKIP  {page_tree.page.source_file} | no chunks")
        return 0

    embed_and_upsert(chunks, client=client, model=model)

    print(
        f"DONE  {page_tree.page.source_file} | "
        f"page_id={page_tree.page.page_id} | chunks={len(chunks)}"
    )
    return len(chunks)


def main() -> None:
    if not TUTORIAL_ROOT.exists():
        raise FileNotFoundError(f"Tutorial root not found: {TUTORIAL_ROOT}")

    run_start = time.perf_counter()
    markdown_files = list(iter_markdown_files(TUTORIAL_ROOT))

    if not markdown_files:
        print(f"No markdown files found under {TUTORIAL_ROOT}")
        return

    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    total_pages = 0
    total_chunks = 0

    print("=" * 100)
    print(f"INGEST START | collection={COLLECTION_NAME} | qdrant={QDRANT_URL}")
    print(f"TUTORIAL ROOT: {TUTORIAL_ROOT}")
    print(f"PAGES FOUND: {len(markdown_files)}")
    print("=" * 100)

    for markdown_path in markdown_files:
        page_start = time.perf_counter()
        chunk_count = ingest_page(markdown_path, client=client, model=model)
        page_elapsed = time.perf_counter() - page_start

        total_pages += 1
        total_chunks += chunk_count

        print(f"TIME  {markdown_path.relative_to(CORPUS_ROOT)} | {page_elapsed:.4f}s")

    total_elapsed = time.perf_counter() - run_start
    print()
    print("=" * 100)
    print(f"INGEST COMPLETE | pages={total_pages} | chunks={total_chunks}")
    print(f"TOTAL TIME: {total_elapsed:.4f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()
