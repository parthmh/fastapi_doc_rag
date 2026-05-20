from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json

import pyromark


CORPUS_ROOT = Path("corpus")
MARKDOWN_PATH = CORPUS_ROOT / "tutorial" / "first-steps.md"
OUTPUT_DIR = Path("processed")


@dataclass
class PageMetadata:
    page_id: str
    page_uid: str
    page_title: str
    page_url: str
    source_file: str


@dataclass
class Node:
    node_id: str
    parent_id: str | None
    level: int
    heading_raw_text: str
    heading_text: str
    anchor_id: str | None
    section_url: str
    heading_inline_code: list[str] = field(default_factory=list)
    section_text_parts: list[str] = field(default_factory=list)
    section_text: str = ""
    section_inline_code: list[str] = field(default_factory=list)
    blocks: list[dict] = field(default_factory=list)
    children: list[Node] = field(default_factory=list)


@dataclass
class FlatNode:
    node_id: str
    parent_id: str | None
    level: int
    heading_raw_text: str
    heading_text: str
    anchor_id: str | None
    section_url: str
    heading_inline_code: list[str] = field(default_factory=list)
    section_text: str = ""
    section_inline_code: list[str] = field(default_factory=list)


@dataclass
class PageTree:
    page: PageMetadata
    roots: list[Node]
    flat_nodes: list[FlatNode]
    nodes_by_id: dict[str, FlatNode]


def normalize_path(path: Path, corpus_root: Path) -> str:
    return path.resolve().relative_to(corpus_root.resolve()).as_posix()


def make_page_id(markdown_path: Path, corpus_root: Path) -> str:
    return Path(normalize_path(markdown_path, corpus_root)).with_suffix("").as_posix()


def make_page_uid(page_id: str) -> str:
    return hashlib.sha1(page_id.encode("utf-8")).hexdigest()


def make_page_url(page_id: str) -> str:
    return f"https://fastapi.tiangolo.com/{page_id}/"


def extract_page_title_from_h1(raw_h1_text: str, fallback_title: str) -> str:
    title = raw_h1_text.split("{ #", 1)[0].strip()
    return title if title else fallback_title


def clean_heading_text(raw_heading_text: str) -> str:
    text = raw_heading_text.split("{ #", 1)[0].strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text


def extract_anchor(raw_heading_text: str) -> str | None:
    marker = "{ #"
    if marker not in raw_heading_text or not raw_heading_text.rstrip().endswith("}"):
        return None
    tail = raw_heading_text.rsplit(marker, 1)[1]
    return tail[:-1].strip() or None


def dedupe_stable(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def build_page(markdown_text: str, markdown_path: Path) -> PageTree:
    page_id = make_page_id(markdown_path, CORPUS_ROOT)
    page = PageMetadata(
        page_id=page_id,
        page_uid=make_page_uid(page_id),
        page_title=page_id.split("/")[-1].replace("-", " ").title(),
        page_url=make_page_url(page_id),
        source_file=normalize_path(markdown_path, CORPUS_ROOT),
    )

    roots: list[Node] = []
    flat_nodes: list[FlatNode] = []
    nodes_by_id: dict[str, FlatNode] = {}
    stack: list[Node] = []

    in_heading = False
    heading_level = 0
    heading_parts: list[str] = []
    heading_inline_code: list[str] = []
    page_title_set = False
    section_index = 0

    in_paragraph = False
    paragraph_parts: list[str] = []
    paragraph_inline_code: list[str] = []

    def sync_node(current_node: Node) -> None:
        current_node.section_text = "".join(current_node.section_text_parts).strip()
        current_node.section_inline_code = dedupe_stable(current_node.section_inline_code)
        flat = nodes_by_id[current_node.node_id]
        flat.section_text = current_node.section_text
        flat.section_inline_code = current_node.section_inline_code.copy()

    def flush_paragraph(current_node: Node | None) -> None:
        nonlocal in_paragraph, paragraph_parts, paragraph_inline_code
        if not in_paragraph or current_node is None:
            return

        text = " ".join(part.strip() for part in paragraph_parts if part.strip()).strip()
        if text:
            current_node.section_text_parts.append(text)
            current_node.section_inline_code.extend(paragraph_inline_code)
            sync_node(current_node)

        in_paragraph = False
        paragraph_parts = []
        paragraph_inline_code = []

    for event in pyromark.events(markdown_text):
        match event:
            case {"Start": {"Heading": {"level": level}}}:
                current_node = stack[-1] if stack else None
                flush_paragraph(current_node)
                in_heading = True
                heading_level = int(str(level)[1:])
                heading_parts = []
                heading_inline_code = []
                continue

            case {"Text": text} if in_heading:
                heading_parts.append(str(text))
                continue

            case {"Code": code} if in_heading:
                code_value = str(code)
                heading_parts.append(code_value)
                heading_inline_code.append(code_value)
                continue

            case {"End": {"Heading": _}} if in_heading:
                raw_heading = "".join(heading_parts).strip()
                heading_text = clean_heading_text(raw_heading)
                anchor_id = extract_anchor(raw_heading)

                if heading_level == 1 and not page_title_set:
                    page.page_title = extract_page_title_from_h1(raw_heading, page.page_title)
                    page_title_set = True

                while stack and stack[-1].level >= heading_level:
                    stack.pop()

                parent_id = stack[-1].node_id if stack else None
                section_index += 1
                node_id = f"{page.page_id}::section-{section_index:04d}"
                section_url = f"{page.page_url}#{anchor_id}" if anchor_id else page.page_url

                node = Node(
                    node_id=node_id,
                    parent_id=parent_id,
                    level=heading_level,
                    heading_raw_text=raw_heading,
                    heading_text=heading_text,
                    anchor_id=anchor_id,
                    section_url=section_url,
                    heading_inline_code=heading_inline_code.copy(),
                )

                flat_node = FlatNode(
                    node_id=node_id,
                    parent_id=parent_id,
                    level=heading_level,
                    heading_raw_text=raw_heading,
                    heading_text=heading_text,
                    anchor_id=anchor_id,
                    section_url=section_url,
                    heading_inline_code=heading_inline_code.copy(),
                )

                if stack:
                    stack[-1].children.append(node)
                else:
                    roots.append(node)

                stack.append(node)
                flat_nodes.append(flat_node)
                nodes_by_id[node_id] = flat_node

                in_heading = False
                heading_level = 0
                heading_parts = []
                heading_inline_code = []
                continue

            case {"Start": "Paragraph"}:
                in_paragraph = True
                paragraph_parts = []
                paragraph_inline_code = []
                continue

            case {"Text": text} if in_paragraph:
                paragraph_parts.append(str(text))
                continue

            case {"Code": code} if in_paragraph:
                code_value = str(code)
                paragraph_parts.append(code_value)
                paragraph_inline_code.append(code_value)
                continue

            case {"End": "Paragraph"} if in_paragraph:
                current_node = stack[-1] if stack else None
                flush_paragraph(current_node)
                continue

    return PageTree(page=page, roots=roots, flat_nodes=flat_nodes, nodes_by_id=nodes_by_id)


def main() -> None:
    markdown_text = MARKDOWN_PATH.read_text(encoding="utf-8")
    tree = build_page(markdown_text, MARKDOWN_PATH)

    print(json.dumps(asdict(tree.page), indent=2, ensure_ascii=False))
    print()
    print(f"ROOTS: {len(tree.roots)}")
    print(f"FLAT: {len(tree.flat_nodes)}")
    print(f"INDEXED: {len(tree.nodes_by_id)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "first_steps_page.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(tree.page), f, indent=2, ensure_ascii=False)

    with (OUTPUT_DIR / "first_steps_tree.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(tree), f, indent=2, ensure_ascii=False)

    with (OUTPUT_DIR / "first_steps_nodes_by_id.json").open("w", encoding="utf-8") as f:
        json.dump({node_id: asdict(node) for node_id, node in tree.nodes_by_id.items()}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
