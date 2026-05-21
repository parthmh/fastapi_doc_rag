from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
from typing import TypedDict

import pyromark


CORPUS_ROOT = Path("corpus")
MARKDOWN_PATH = CORPUS_ROOT / "tutorial" / "first-steps.md"
OUTPUT_DIR = Path("processed")
SECTION_SEPARATOR = "\n\n"
LINE_FEED = "\n"


class LinkRef(TypedDict):
    text: str
    url: str
    section_id: str
    section_url: str


class CodeRef(TypedDict):
    type: str
    path: str
    highlight_lines: list[int]
    raw: str
    section_id: str
    section_url: str


class CodeBlock(TypedDict):
    type: str
    language: str | None
    text: str
    section_id: str
    section_url: str


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
    node_kind: str
    heading_text: str
    anchor_id: str | None
    section_url: str
    section_text_parts: list[str] = field(default_factory=list)
    section_text: str = ""
    section_inline_code: list[str] = field(default_factory=list)
    section_links: list[LinkRef] = field(default_factory=list)
    code_refs: list[CodeRef] = field(default_factory=list)
    code_blocks: list[CodeBlock] = field(default_factory=list)
    admonition_kind: str | None = None
    admonition_title: str | None = None
    children: list[Node] = field(default_factory=list)


@dataclass
class FlatNode:
    node_id: str
    parent_id: str | None
    level: int
    node_kind: str
    heading_text: str
    anchor_id: str | None
    section_url: str
    section_text: str = ""
    section_inline_code: list[str] = field(default_factory=list)
    section_links: list[LinkRef] = field(default_factory=list)
    code_refs: list[CodeRef] = field(default_factory=list)
    code_blocks: list[CodeBlock] = field(default_factory=list)
    admonition_kind: str | None = None
    admonition_title: str | None = None


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


def parse_admonition_start(text: str) -> tuple[str, str | None] | None:
    stripped = text.strip()
    if not stripped.startswith("///") or stripped == "///":
        return None

    tail = stripped[3:].strip()
    if not tail:
        return "admonition", None

    if "|" in tail:
        kind_part, title_part = tail.split("|", 1)
        kind = kind_part.strip() or "admonition"
        title = title_part.strip() or None
        return kind, title

    parts = tail.split(None, 1)
    kind = parts[0].strip() or "admonition"
    title = parts[1].strip() if len(parts) > 1 else None
    return kind, title


def parse_code_ref(text: str) -> CodeRef | None:
    stripped = text.strip()
    if not (stripped.startswith("{*") and stripped.endswith("*}")):
        return None

    inner = stripped[2:-2].strip()
    if not inner:
        return None

    raw_path = inner
    highlight_lines: list[int] = []

    if "hl[" in inner:
        path_part, hl_part = inner.split("hl[", 1)
        raw_path = path_part.strip()
        if hl_part.endswith("]"):
            hl_spec = hl_part[:-1].strip()
            for chunk in hl_spec.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if "-" in chunk:
                    start_str, end_str = chunk.split("-", 1)
                    highlight_lines.extend(range(int(start_str), int(end_str) + 1))
                else:
                    highlight_lines.append(int(chunk))

    return {
        "type": "code_ref",
        "path": raw_path.replace("\\", "/").strip(),
        "highlight_lines": highlight_lines,
        "raw": stripped,
        "section_id": "",
        "section_url": "",
    }


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
    heading_stack: list[Node] = []

    in_heading = False
    heading_level = 0
    heading_parts: list[str] = []
    heading_inline_code: list[str] = []
    page_title_set = False
    section_index = 0
    admonition_index = 0

    in_paragraph = False
    paragraph_parts: list[str] = []
    paragraph_inline_code: list[str] = []

    in_admonition = False
    current_admonition_node: Node | None = None

    in_code_block = False
    code_block_parts: list[str] = []
    current_code_block: CodeBlock | None = None

    in_link = False
    link_url: str | None = None
    link_parts: list[str] = []

    def sync_node(current_node: Node) -> None:
        current_node.section_text = SECTION_SEPARATOR.join(current_node.section_text_parts).strip()
        current_node.section_inline_code = dedupe_stable(current_node.section_inline_code)

        flat = nodes_by_id[current_node.node_id]
        flat.section_text = current_node.section_text
        flat.section_inline_code = current_node.section_inline_code.copy()
        flat.section_links = [link.copy() for link in current_node.section_links]
        flat.code_refs = [ref.copy() for ref in current_node.code_refs]
        flat.code_blocks = [block.copy() for block in current_node.code_blocks]
        flat.admonition_kind = current_node.admonition_kind
        flat.admonition_title = current_node.admonition_title

    def append_text(current_node: Node | None, text: str, inline_code: list[str]) -> None:
        if current_node is None:
            return
        cleaned = text.strip()
        if not cleaned:
            return
        current_node.section_text_parts.append(cleaned)
        current_node.section_inline_code.extend(inline_code)
        sync_node(current_node)

    def flush_paragraph(current_node: Node | None) -> None:
        nonlocal in_paragraph, paragraph_parts, paragraph_inline_code
        if not in_paragraph:
            return
        paragraph_text = " ".join(part.strip() for part in paragraph_parts if part.strip()).strip()
        append_text(current_node, paragraph_text, paragraph_inline_code)
        in_paragraph = False
        paragraph_parts = []
        paragraph_inline_code = []

    def add_flat_node(node: Node) -> None:
        flat = FlatNode(
            node_id=node.node_id,
            parent_id=node.parent_id,
            level=node.level,
            node_kind=node.node_kind,
            heading_text=node.heading_text,
            anchor_id=node.anchor_id,
            section_url=node.section_url,
            section_text=node.section_text,
            section_inline_code=node.section_inline_code.copy(),
            section_links=[link.copy() for link in node.section_links],
            code_refs=[ref.copy() for ref in node.code_refs],
            code_blocks=[block.copy() for block in node.code_blocks],
            admonition_kind=node.admonition_kind,
            admonition_title=node.admonition_title,
        )
        flat_nodes.append(flat)
        nodes_by_id[node.node_id] = flat

    def start_heading_section(heading_text: str, raw_heading: str, anchor_id: str | None, level: int, heading_code: list[str]) -> None:
        nonlocal section_index, page_title_set

        while heading_stack and heading_stack[-1].level >= level:
            heading_stack.pop()

        parent_id = heading_stack[-1].node_id if heading_stack else None
        section_index += 1
        node_id = f"{page.page_id}::section-{section_index:04d}"
        section_url = f"{page.page_url}#{anchor_id}" if anchor_id else page.page_url

        if level == 1 and not page_title_set:
            page.page_title = extract_page_title_from_h1(raw_heading, page.page_title)
            page_title_set = True

        node = Node(
            node_id=node_id,
            parent_id=parent_id,
            level=level,
            node_kind="heading",
            heading_text=heading_text,
            anchor_id=anchor_id,
            section_url=section_url,
        )
        node.section_inline_code.extend(heading_code)

        if heading_stack:
            heading_stack[-1].children.append(node)
        else:
            roots.append(node)

        heading_stack.append(node)
        add_flat_node(node)

    def start_admonition_section(kind: str, title: str | None) -> None:
        nonlocal admonition_index, current_admonition_node, in_admonition

        parent = heading_stack[-1] if heading_stack else None
        if parent is None:
            return

        admonition_index += 1
        node_id = f"{parent.node_id}::admonition-{admonition_index:04d}"
        node = Node(
            node_id=node_id,
            parent_id=parent.node_id,
            level=parent.level + 1,
            node_kind="admonition",
            heading_text=title or kind,
            anchor_id=None,
            section_url=parent.section_url,
            admonition_kind=kind,
            admonition_title=title,
        )
        parent.children.append(node)
        current_admonition_node = node
        in_admonition = True
        add_flat_node(node)

    def current_target_node() -> Node | None:
        if in_admonition and current_admonition_node is not None:
            return current_admonition_node
        return heading_stack[-1] if heading_stack else None

    for event in pyromark.events(markdown_text):
        match event:
            case {"Start": {"Heading": {"level": level}}}:
                flush_paragraph(current_target_node())
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
                start_heading_section(heading_text, raw_heading, anchor_id, heading_level, heading_inline_code)
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

            case {"Text": text} if in_paragraph and not in_link:
                paragraph_parts.append(str(text))
                continue

            case {"Text": text} if in_paragraph and in_link:
                paragraph_parts.append(str(text))
                link_parts.append(str(text))
                continue

            case {"Code": code} if in_paragraph:
                code_value = str(code)
                paragraph_parts.append(code_value)
                paragraph_inline_code.append(code_value)
                continue

            case {"Start": {"Link": {"dest_url": url}}} if in_paragraph:
                in_link = True
                link_url = str(url)
                link_parts = []
                continue

            case {"End": "Link"} if in_paragraph and in_link:
                current_node = current_target_node()
                link_text = "".join(link_parts).strip()
                if current_node is not None and link_url and link_text:
                    link_ref: LinkRef = {
                        "text": link_text,
                        "url": link_url,
                        "section_id": current_node.node_id,
                        "section_url": current_node.section_url,
                    }
                    current_node.section_links.append(link_ref)
                    sync_node(current_node)
                in_link = False
                link_url = None
                link_parts = []
                continue

            case {"End": "Paragraph"} if in_paragraph:
                current_node = current_target_node()
                paragraph_text = " ".join(part.strip() for part in paragraph_parts if part.strip()).strip()

                code_ref = parse_code_ref(paragraph_text)
                if code_ref is not None:
                    if current_node is not None:
                        code_ref["section_id"] = current_node.node_id
                        code_ref["section_url"] = current_node.section_url
                        current_node.code_refs.append(code_ref)
                        sync_node(current_node)
                else:
                    admonition_start = parse_admonition_start(paragraph_text)
                    if admonition_start is not None and not in_admonition:
                        kind, title = admonition_start
                        start_admonition_section(kind, title)
                    elif in_admonition and paragraph_text == "///":
                        current_admonition_node = None
                        in_admonition = False
                    else:
                        append_text(current_node, paragraph_text, paragraph_inline_code)

                in_paragraph = False
                paragraph_parts = []
                paragraph_inline_code = []
                continue

            case {"Start": {"CodeBlock": {"Fenced": fenced}}}:
                current_node = current_target_node()
                if current_node is not None:
                    in_code_block = True
                    code_block_parts = []
                    current_code_block = {
                        "type": "code_block",
                        "language": str(fenced) or None,
                        "text": "",
                        "section_id": current_node.node_id,
                        "section_url": current_node.section_url,
                    }
                continue

            case {"Text": text} if in_code_block:
                code_block_parts.append(str(text))
                continue

            case {"End": "CodeBlock"} if in_code_block:
                current_node = current_target_node()
                if current_node is not None and current_code_block is not None:
                    current_code_block["text"] = LINE_FEED.join(code_block_parts).rstrip(LINE_FEED)
                    current_node.code_blocks.append(current_code_block)
                    sync_node(current_node)
                in_code_block = False
                code_block_parts = []
                current_code_block = None
                continue

            case {"Start": "HtmlBlock"}:
                continue

            case {"End": "HtmlBlock"}:
                continue

            case _:
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
