from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any, Iterable

from .chunk_markdown import build_chunks_from_page_tree
from .parse_markdown import AdmonitionNode, FlatNode, PageMetadata, PageTree


DEBUG_PARSE_ROOT = Path("processed") / "debug"
DEBUG_CHUNK_ROOT = Path("processed") / "debug_chunks"
SAMPLE_OUTPUT_PATH = Path("processed") / "debug" / "sample_chunks.json"


def iter_debug_page_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return

    yield from sorted(
        path for path in root.iterdir() if path.is_dir()
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_page_tree(debug_page_dir: Path) -> PageTree:
    page_raw = load_json(debug_page_dir / "page.json")
    headings_raw = load_json(debug_page_dir / "headings_by_id.json")
    admonitions_raw = load_json(debug_page_dir / "admonitions_by_id.json")

    page = PageMetadata(**page_raw)

    flat_nodes = [
        FlatNode(**node)
        for node in headings_raw.values()
    ]

    admonition_nodes = [
        AdmonitionNode(**node)
        for node in admonitions_raw.values()
    ]

    nodes_by_id = {
        node.node_id: node
        for node in flat_nodes
    }

    admonition_nodes_by_id = {
        node.node_id: node
        for node in admonition_nodes
    }

    return PageTree(
        page=page,
        roots=[],
        flat_nodes=flat_nodes,
        nodes_by_id=nodes_by_id,
        admonition_nodes=admonition_nodes,
        admonition_nodes_by_id=admonition_nodes_by_id,
        section_timings=[],
    )


def write_chunks_for_page(page_id_slug: str, chunks: list[dict[str, Any]]) -> Path:
    page_dir = DEBUG_CHUNK_ROOT / page_id_slug
    page_dir.mkdir(parents=True, exist_ok=True)

    output_path = page_dir / "chunks.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return output_path


def write_sample_chunks(chunks: list[dict[str, Any]]) -> None:
    SAMPLE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SAMPLE_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks[:10], f, indent=2, ensure_ascii=False)


def main() -> None:
    if not DEBUG_PARSE_ROOT.exists():
        raise FileNotFoundError(
            f"{DEBUG_PARSE_ROOT} not found. "
            f"Run run_parse_markdown.py first."
        )

    run_start = time.perf_counter()
    total_pages = 0
    total_chunks = 0
    sample_chunks: list[dict[str, Any]] = []

    for debug_page_dir in iter_debug_page_dirs(DEBUG_PARSE_ROOT):
        page_tree = load_page_tree(debug_page_dir)

        chunk_start = time.perf_counter()
        chunks = build_chunks_from_page_tree(page_tree)
        chunk_elapsed = time.perf_counter() - chunk_start

        total_pages += 1
        total_chunks += len(chunks)

        if not sample_chunks:
            sample_chunks = chunks[:10]

        output_path = write_chunks_for_page(
            page_id_slug=debug_page_dir.name,
            chunks=chunks,
        )

        print()
        print("=" * 100)
        print(f"CHUNKED PAGE: {page_tree.page.source_file}")
        print(f"PAGE ID: {page_tree.page.page_id}")
        print(f"CHUNKS: {len(chunks)}")
        print(f"CHUNK TIME: {chunk_elapsed:.4f}s")
        print(f"WROTE: {output_path}")

    write_sample_chunks(sample_chunks)

    total_elapsed = time.perf_counter() - run_start
    print()
    print("=" * 100)
    print(f"CHUNKED PAGES: {total_pages}")
    print(f"TOTAL CHUNKS: {total_chunks}")
    print(f"TOTAL TIME: {total_elapsed:.4f}s")
    print(f"SAMPLE OUTPUT: {SAMPLE_OUTPUT_PATH}")
    print(f"CHUNK DEBUG ROOT: {DEBUG_CHUNK_ROOT}")


if __name__ == "__main__":
    main()