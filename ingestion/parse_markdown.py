from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import time
from typing import TypedDict

import pyromark


CORPUS_ROOT = Path("corpus")
TUTORIAL_ROOT = CORPUS_ROOT/ "tutorial"

OUTPUT_DIR = Path("processed")
DEBUG_OUTPUT_DIR = Path("processed/debug")

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


class SectionTiming(TypedDict):
    node_id: str
    seconds: float
    paragraphs: int
    list_items: int
    code_blocks: int
    code_refs: int
    links: int
    admonitions: int


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


@dataclass
class AdmonitionNode:
    node_id: str
    parent_id: str
    node_kind: str
    kind: str
    title: str | None
    section_url: str
    fence_depth: int
    section_text_parts: list[str] = field(default_factory=list)
    section_text: str = ""
    section_inline_code: list[str] = field(default_factory=list)
    section_links: list[LinkRef] = field(default_factory=list)
    code_refs: list[CodeRef] = field(default_factory=list)
    code_blocks: list[CodeBlock] = field(default_factory=list)


@dataclass
class PageTree:
    page: PageMetadata
    roots: list[Node]
    flat_nodes: list[FlatNode]
    nodes_by_id: dict[str, FlatNode]
    admonition_nodes: list[AdmonitionNode] = field(default_factory=list)
    admonition_nodes_by_id: dict[str, AdmonitionNode] = field(default_factory=dict)
    section_timings: list[SectionTiming] = field(default_factory=list)


def write_debug_page(page_tree: PageTree) -> None:
    page_id = page_tree.page.page_id.replace("/", "__")

    page_dir = DEBUG_OUTPUT_DIR / page_id
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
            {node_id: asdict(node) for node_id, node in page_tree.admonition_nodes_by_id.items()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (page_dir / "section_timings.json").write_text(
        json.dumps(page_tree.section_timings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def iter_markdown_files(root:Path):
    yield from sorted(root.rglob("*.md"))



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


def parse_admonition_start(text: str) -> tuple[int, str, str | None, bool] | None:
    stripped = text.strip()

    if not stripped.startswith("/"):
        return None

    fence_depth = len(stripped) - len(stripped.lstrip("/"))
    if fence_depth < 3:
        return None

    tail = stripped[fence_depth:].strip()

    # Bare fence line, usually a closer
    if not tail:
        return fence_depth, "admonition", None, False

    if "|" in tail:
        kind_part, title_part = tail.split("|", 1)
        kind = kind_part.strip() or "admonition"
        title = title_part.strip() or None
        return fence_depth, kind, title, True

    parts = tail.split(None, 1)
    kind = parts[0].strip() or "admonition"
    title = parts[1].strip() if len(parts) > 1 else None
    return fence_depth, kind, title, True


def parse_code_ref(text: str) -> CodeRef | None:
    stripped = text.strip()
    if not (stripped.startswith("{*") and stripped.endswith("*}")):
        return None

    inner = stripped[2:-2].strip()
    if not inner:
        return None

    # The actual path is always the first token.
    raw_path = inner.split()[0]
    highlight_lines: list[int] = []

    hl_start = inner.find("hl[")
    if hl_start != -1:
        after_hl = inner[hl_start + 3 :]
        hl_end = after_hl.find("]")
        if hl_end != -1:
            hl_spec = after_hl[:hl_end].strip()

            for chunk in hl_spec.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue

                if "-" in chunk:
                    start_str, end_str = chunk.split("-", 1)
                    highlight_lines.extend(range(int(start_str), int(end_str) + 1))
                elif ":" in chunk:
                    start_str, end_str = chunk.split(":", 1)
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
    admonition_nodes: list[AdmonitionNode] = []
    admonition_nodes_by_id: dict[str, AdmonitionNode] = {}
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

    in_list = False
    in_item = False
    item_parts: list[str] = []
    item_inline_code: list[str] = []

    admonition_stack: list[AdmonitionNode] = []

    in_code_block = False
    code_block_parts: list[str] = []
    current_code_block: CodeBlock | None = None

    in_link = False
    link_url: str | None = None
    link_parts: list[str] = []

    current_section_id: str | None = None
    current_section_start: float | None = None
    current_section_events: dict[str, int] = {
        "paragraphs": 0,
        "list_items": 0,
        "code_blocks": 0,
        "code_refs": 0,
        "links": 0,
        "admonitions": 0,
    }
    section_timings: list[SectionTiming] = []

    def close_current_section() -> None:
        nonlocal current_section_id, current_section_start, current_section_events
        if current_section_id is None or current_section_start is None:
            return

        section_timings.append(
            {
                "node_id": current_section_id,
                "seconds": time.perf_counter() - current_section_start,
                "paragraphs": current_section_events["paragraphs"],
                "list_items": current_section_events["list_items"],
                "code_blocks": current_section_events["code_blocks"],
                "code_refs": current_section_events["code_refs"],
                "links": current_section_events["links"],
                "admonitions": current_section_events["admonitions"],
            }
        )

        current_section_id = None
        current_section_start = None
        current_section_events = {
            "paragraphs": 0,
            "list_items": 0,
            "code_blocks": 0,
            "code_refs": 0,
            "links": 0,
            "admonitions": 0,
        }

    def sync_heading(node: Node) -> None:
        node.section_inline_code = dedupe_stable(node.section_inline_code)
        flat = nodes_by_id[node.node_id]
        flat.section_text = node.section_text
        flat.section_inline_code = node.section_inline_code
        flat.section_links = node.section_links
        flat.code_refs = node.code_refs
        flat.code_blocks = node.code_blocks

    def sync_admonition(node: AdmonitionNode) -> None:
        node.section_inline_code = dedupe_stable(node.section_inline_code)
        admonition_nodes_by_id[node.node_id] = node

    def append_text(node: Node | AdmonitionNode | None, text: str, inline_code: list[str]) -> None:
        if node is None:
            return
        cleaned = text.strip()
        if not cleaned:
            return
        node.section_text_parts.append(cleaned)
        if node.section_text:
            node.section_text = f"{node.section_text}{SECTION_SEPARATOR}{cleaned}"
        else:
            node.section_text = cleaned
        node.section_inline_code.extend(inline_code)
        if isinstance(node, AdmonitionNode):
            sync_admonition(node)
        else:
            sync_heading(node)

    def flush_paragraph(node: Node | AdmonitionNode | None) -> None:
        nonlocal in_paragraph, paragraph_parts, paragraph_inline_code
        if not in_paragraph:
            return
        paragraph_text = " ".join(part.strip() for part in paragraph_parts if part.strip()).strip()
        append_text(node, paragraph_text, paragraph_inline_code)
        in_paragraph = False
        paragraph_parts = []
        paragraph_inline_code = []

    def flush_list_item(node: Node | AdmonitionNode | None) -> None:
        nonlocal in_item, item_parts, item_inline_code
        if not in_item:
            return
        item_text = " ".join(part.strip() for part in item_parts if part.strip()).strip()
        if item_text:
            append_text(node, f"- {item_text}", item_inline_code)
        in_item = False
        item_parts = []
        item_inline_code = []


    def start_heading_section(heading_text: str, raw_heading: str, anchor_id: str | None, level: int, heading_code: list[str]) -> None:
        nonlocal section_index, page_title_set, current_section_id, current_section_start

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
        flat = FlatNode(
            node_id=node.node_id,
            parent_id=node.parent_id,
            level=node.level,
            node_kind=node.node_kind,
            heading_text=node.heading_text,
            anchor_id=node.anchor_id,
            section_url=node.section_url,
            section_text=node.section_text,
            section_inline_code=node.section_inline_code,
            section_links=node.section_links,
            code_refs=node.code_refs,
            code_blocks=node.code_blocks,
        )
        flat_nodes.append(flat)
        nodes_by_id[node.node_id] = flat

        close_current_section()
        current_section_id = node_id
        current_section_start = time.perf_counter()

    def start_admonition_section(fence_depth: int, kind: str, title: str | None) -> None:
        nonlocal admonition_index

        parent = admonition_stack[-1] if admonition_stack else (heading_stack[-1] if heading_stack else None)
        if parent is None:
            return

        admonition_index += 1
        node_id = f"{parent.node_id}::admonition-{admonition_index:04d}"
        node = AdmonitionNode(
            node_id=node_id,
            parent_id=parent.node_id,
            node_kind="admonition",
            kind=kind,
            title=title,
            section_url=parent.section_url,
            fence_depth=fence_depth,
        )
        admonition_nodes.append(node)
        admonition_nodes_by_id[node.node_id] = node
        admonition_stack.append(node)

    def current_target_node() -> Node | AdmonitionNode | None:
        if admonition_stack:
            return admonition_stack[-1]
        return heading_stack[-1] if heading_stack else None

    for event in pyromark.events(markdown_text):
        match event:
            case {"Start": {"Heading": {"level": level}}}:
                flush_paragraph(current_target_node())
                flush_list_item(current_target_node())
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
                    current_section_events["links"] += 1
                    if isinstance(current_node, AdmonitionNode):
                        sync_admonition(current_node)
                    else:
                        sync_heading(current_node)
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
                        current_section_events["code_refs"] += 1
                        if isinstance(current_node, AdmonitionNode):
                            sync_admonition(current_node)
                        else:
                            sync_heading(current_node)
                else:
                    admonition_start = parse_admonition_start(paragraph_text)

                    if admonition_start is not None:
                        fence_depth, kind, title, is_opening = admonition_start

                        # Closing fence:
                        # - bare fence line
                        # - or a fence with same depth as the current open admonition
                        if not is_opening:
                            if admonition_stack:
                                sync_admonition(admonition_stack[-1])
                                admonition_stack.pop()
                                current_section_events["admonitions"] += 1
                        else:
                            # Opening fence
                            start_admonition_section(fence_depth, kind, title)
                    else:
                        append_text(current_node, paragraph_text, paragraph_inline_code)
                        current_section_events["paragraphs"] += 1

                in_paragraph = False
                paragraph_parts = []
                paragraph_inline_code = []
                continue

            case {"Start": {"List": None}}:
                in_list = True
                continue

            case {"Start": "Item"} if in_list:
                in_item = True
                item_parts = []
                item_inline_code = []
                continue

            case {"Text": text} if in_item and not in_link:
                item_parts.append(str(text))
                continue

            case {"Text": text} if in_item and in_link:
                item_parts.append(str(text))
                link_parts.append(str(text))
                continue

            case {"Code": code} if in_item:
                code_value = str(code)
                item_parts.append(code_value)
                item_inline_code.append(code_value)
                continue

            case {"Start": {"Link": {"dest_url": url}}} if in_item:
                in_link = True
                link_url = str(url)
                link_parts = []
                continue

            case {"End": "Link"} if in_item and in_link:
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
                    current_section_events["links"] += 1
                    if isinstance(current_node, AdmonitionNode):
                        sync_admonition(current_node)
                    else:
                        sync_heading(current_node)
                in_link = False
                link_url = None
                link_parts = []
                continue

            case {"End": "Item"} if in_item:
                current_node = current_target_node()
                item_text = " ".join(part.strip() for part in item_parts if part.strip()).strip()
                append_text(current_node, f"- {item_text}" if item_text else "", item_inline_code)
                if item_text:
                    current_section_events["list_items"] += 1
                in_item = False
                item_parts = []
                item_inline_code = []
                continue

            case {"End": {"List": False}} if in_list:
                in_list = False
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
                    current_section_events["code_blocks"] += 1
                    if isinstance(current_node, AdmonitionNode):
                        sync_admonition(current_node)
                    else:
                        sync_heading(current_node)
                in_code_block = False
                code_block_parts = []
                current_code_block = None
                continue

            case {"Start": "HtmlBlock"}:
                continue

            case {"End": "HtmlBlock"}:
                continue

            case {"Start": "Strong"}:
                continue

            case {"End": "Strong"}:
                continue

            case {"Start": "Emphasis"}:
                continue

            case {"End": "Emphasis"}:
                continue

            case _:
                continue

    close_current_section()
    return PageTree(
        page=page,
        roots=roots,
        flat_nodes=flat_nodes,
        nodes_by_id=nodes_by_id,
        admonition_nodes=admonition_nodes,
        admonition_nodes_by_id=admonition_nodes_by_id,
        section_timings=section_timings,
    )


def main() -> None:
    run_start = time.perf_counter()

    if not TUTORIAL_ROOT.exists():
        raise FileNotFoundError(f"Tutorial root not found: {TUTORIAL_ROOT}")

    pages = []
    parse_start = time.perf_counter()

    for markdown_path in iter_markdown_files(TUTORIAL_ROOT):
        markdown_text = markdown_path.read_text(encoding="utf-8")
        page_tree = build_page(markdown_text, markdown_path)
        write_debug_page(page_tree)
        pages.append(page_tree)

        print()
        print("=" * 100)
        print(f"PARSED: {page_tree.page.source_file}")
        print(f"PAGE ID: {page_tree.page.page_id}")
        print(f"ROOTS: {len(page_tree.roots)}")
        print(f"FLAT HEADINGS: {len(page_tree.flat_nodes)}")
        print(f"ADMONITIONS: {len(page_tree.admonition_nodes)}")
        print(f"SECTIONS TIMED: {len(page_tree.section_timings)}")

    parse_elapsed = time.perf_counter() - parse_start
    total_elapsed = time.perf_counter() - run_start

    print()
    print("=" * 100)
    print(f"PARSED PAGES: {len(pages)}")
    print(f"PARSE TIME: {parse_elapsed:.4f}s")
    print(f"TOTAL TIME: {total_elapsed:.4f}s")


if __name__ == "__main__":
    main()
