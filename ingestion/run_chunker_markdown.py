from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any, Iterable

from .chunk_markdown import build_chunks_from_page_tree
from .parse_markdown import (
    AdmonitionNode,
    FlatNode,
    Node,
    PageMetadata,
    PageTree,
)

DEBUG_PARSE_ROOT = Path("processed") / "debug"

CHUNK_STRATEGIES = [
    "section",
    "h2_subtree",
]

DEBUG_OUTPUT_ROOT = Path("processed")
DEBUG_SAMPLE_ROOT = DEBUG_OUTPUT_ROOT / "debug"


def iter_debug_page_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return

    yield from sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_node_tree(node_raw: dict[str, Any]) -> Node:
    children = [
        build_node_tree(child)
        for child in node_raw.get("children", [])
    ]

    return Node(
        node_id=node_raw["node_id"],
        parent_id=node_raw.get("parent_id"),
        level=int(node_raw.get("level", 0)),
        node_kind=node_raw.get("node_kind", "heading"),
        heading_text=node_raw.get("heading_text", ""),
        anchor_id=node_raw.get("anchor_id"),
        section_url=node_raw.get("section_url", ""),
        section_text_parts=node_raw.get("section_text_parts", []),
        section_text=node_raw.get("section_text", ""),
        section_inline_code=node_raw.get("section_inline_code", []),
        section_links=node_raw.get("section_links", []),
        code_refs=node_raw.get("code_refs", []),
        code_blocks=node_raw.get("code_blocks", []),
        children=children,
    )


def load_page_tree(debug_page_dir: Path) -> PageTree:
    page_raw = load_json(debug_page_dir / "page.json")
    headings_raw = load_json(debug_page_dir / "headings_by_id.json")
    admonitions_raw = load_json(debug_page_dir / "admonitions_by_id.json")
    tree_raw = load_json(debug_page_dir / "tree.json")

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

    roots = [
        build_node_tree(root_raw)
        for root_raw in tree_raw.get("roots", [])
    ]

    return PageTree(
        page=page,
        roots=roots,
        flat_nodes=flat_nodes,
        nodes_by_id=nodes_by_id,
        admonition_nodes=admonition_nodes,
        admonition_nodes_by_id=admonition_nodes_by_id,
        section_timings=tree_raw.get("section_timings", []),
    )


def write_chunks_for_page(
    strategy: str,
    page_id_slug: str,
    chunks: list[dict[str, Any]],
) -> Path:
    output_root = DEBUG_OUTPUT_ROOT / f"debug_chunks_{strategy}"

    page_dir = output_root / page_id_slug
    page_dir.mkdir(parents=True, exist_ok=True)

    output_path = page_dir / "chunks.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def write_sample_chunks(
    strategy: str,
    chunks: list[dict[str, Any]],
) -> None:
    DEBUG_SAMPLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_path = (
        DEBUG_SAMPLE_ROOT
        / f"sample_chunks_{strategy}.json"
    )

    with sample_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            chunks[:10],
            f,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    if not DEBUG_PARSE_ROOT.exists():
        raise FileNotFoundError(
            f"{DEBUG_PARSE_ROOT} not found. "
            f"Run run_parse_markdown.py first."
        )

    run_start = time.perf_counter()

    total_pages = 0

    strategy_chunk_counts = {
        strategy: 0
        for strategy in CHUNK_STRATEGIES
    }

    sample_written = {
        strategy: False
        for strategy in CHUNK_STRATEGIES
    }

    for debug_page_dir in iter_debug_page_dirs(DEBUG_PARSE_ROOT):
        page_tree = load_page_tree(debug_page_dir)

        total_pages += 1

        print()
        print("=" * 100)
        print(f"PAGE: {page_tree.page.page_id}")

        for strategy in CHUNK_STRATEGIES:
            chunk_start = time.perf_counter()

            chunks = build_chunks_from_page_tree(
                page_tree,
                strategy=strategy,
            )

            elapsed = time.perf_counter() - chunk_start

            strategy_chunk_counts[strategy] += len(chunks)

            output_path = write_chunks_for_page(
                strategy=strategy,
                page_id_slug=debug_page_dir.name,
                chunks=chunks,
            )

            if not sample_written[strategy]:
                write_sample_chunks(
                    strategy,
                    chunks,
                )
                sample_written[strategy] = True

            print()
            print(f"Strategy   : {strategy}")
            print(f"Chunks     : {len(chunks)}")
            print(f"Time       : {elapsed:.4f}s")
            print(f"Output     : {output_path}")

    total_elapsed = time.perf_counter() - run_start

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(f"Pages processed : {total_pages}")

    print()

    for strategy in CHUNK_STRATEGIES:
        print(
            f"{strategy:<15} "
            f"{strategy_chunk_counts[strategy]:>8} chunks"
        )

    print()
    print(f"Total time : {total_elapsed:.4f}s")

    print()
    print("Output directories:")

    for strategy in CHUNK_STRATEGIES:
        print(
            DEBUG_OUTPUT_ROOT
            / f"debug_chunks_{strategy}"
        )


if __name__ == "__main__":
    main()