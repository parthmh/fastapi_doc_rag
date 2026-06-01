from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time
from typing import Iterable

from .parse_markdown import PageTree, build_page


CORPUS_ROOT = Path("corpus")
TUTORIAL_ROOT = CORPUS_ROOT / "tutorial"
DEBUG_OUTPUT_ROOT = Path("processed") / "debug"


def iter_markdown_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.md"))


def write_debug_page(page_tree: PageTree, debug_root: Path = DEBUG_OUTPUT_ROOT) -> None:
    page_id_slug = page_tree.page.page_id.replace("/", "__")
    page_dir = debug_root / page_id_slug
    page_dir.mkdir(parents=True, exist_ok=True)

    (page_dir / "page.json").write_text(
        json.dumps(asdict(page_tree.page), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (page_dir / "tree.json").write_text(
        json.dumps(asdict(page_tree), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (page_dir / "headings_by_id.json").write_text(
        json.dumps(
            {node_id: asdict(node) for node_id, node in page_tree.nodes_by_id.items()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (page_dir / "admonitions_by_id.json").write_text(
        json.dumps(
            {
                node_id: asdict(node)
                for node_id, node in page_tree.admonition_nodes_by_id.items()
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (page_dir / "section_timings.json").write_text(
        json.dumps(page_tree.section_timings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    if not TUTORIAL_ROOT.exists():
        raise FileNotFoundError(f"Tutorial root not found: {TUTORIAL_ROOT}")

    run_start = time.perf_counter()
    page_count = 0

    for markdown_path in iter_markdown_files(TUTORIAL_ROOT):
        markdown_text = markdown_path.read_text(encoding="utf-8")

        parse_start = time.perf_counter()
        page_tree = build_page(markdown_text, markdown_path)
        parse_elapsed = time.perf_counter() - parse_start

        write_debug_page(page_tree)
        page_count += 1

        print()
        print("=" * 100)
        print(f"PARSED: {page_tree.page.source_file}")
        print(f"PAGE ID: {page_tree.page.page_id}")
        print(f"ROOTS: {len(page_tree.roots)}")
        print(f"FLAT HEADINGS: {len(page_tree.flat_nodes)}")
        print(f"ADMONITIONS: {len(page_tree.admonition_nodes)}")
        print(f"SECTIONS TIMED: {len(page_tree.section_timings)}")
        print(f"PARSE TIME: {parse_elapsed:.4f}s")

    total_elapsed = time.perf_counter() - run_start
    print()
    print("=" * 100)
    print(f"PARSED PAGES: {page_count}")
    print(f"TOTAL TIME: {total_elapsed:.4f}s")
    print(f"DEBUG OUTPUT ROOT: {DEBUG_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
