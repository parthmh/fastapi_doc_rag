from __future__ import annotations

from typing import Any, Iterable
import re

from chonkie import SentenceChunker
from transformers import AutoTokenizer

from .parse_markdown import AdmonitionNode, FlatNode, Node, PageTree


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 230
CHUNK_OVERLAP = 24

SECTION_SEPARATOR = "\n\n"

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
CHUNKER = SentenceChunker(
    tokenizer=TOKENIZER,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    min_sentences_per_chunk=1,
)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def heading_text(node: FlatNode) -> str:
    heading = clean_text(node.heading_text or "")
    body = clean_text(node.section_text or "")
    if heading and body:
        return f"{heading}. {body}"
    return heading or body


def admonition_text(node: AdmonitionNode) -> str:
    title = clean_text(node.title or "")
    body = clean_text(node.section_text or "")
    if title and body:
        return f"{title}. {body}"
    return title or body


def collect_node_text(node: Node) -> str:
    """
    Collect text for a heading node without descendants.
    """
    heading = clean_text(node.heading_text or "")
    body = clean_text(node.section_text or "")
    if heading and body:
        return f"{heading}. {body}"
    return heading or body


def collect_subtree_text(node: Node) -> str:
    """
    Collect text for a heading node plus all descendant heading nodes.
    """
    parts: list[str] = []

    own_text = collect_node_text(node)
    if own_text:
        parts.append(own_text)

    for child in node.children:
        child_text = collect_subtree_text(child)
        if child_text:
            parts.append(child_text)

    return SECTION_SEPARATOR.join(parts)


def chunk_heading_node(
    node: Node | FlatNode,
    page_id: str,
    chunk_kind: str,
    text: str,
) -> list[dict[str, Any]]:
    chunks = CHUNKER.chunk(text)
    output: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        chunk_text = clean_text(getattr(chunk, "text", str(chunk)))
        token_count = int(
            getattr(chunk, "token_count", 0)
            or len(TOKENIZER.encode(chunk_text, add_special_tokens=False))
        )

        output.append(
            {
                "chunk_id": f"{node.node_id}::chunk-{idx:04d}",
                "chunk_kind": chunk_kind,
                "source_node_id": node.node_id,
                "parent_id": node.parent_id,
                "page_id": page_id,
                "node_kind": node.node_kind,
                "level": int(getattr(node, "level", 0)),
                "heading_text": getattr(node, "heading_text", ""),
                "section_url": node.section_url,
                "chunk_index": idx,
                "chunk_text": chunk_text,
                "token_count": token_count,
                "kind": getattr(node, "kind", None),
                "title": getattr(node, "title", None),
            }
        )

    return output


def chunk_admonition_node(
    node: AdmonitionNode,
    page_id: str,
    text: str,
) -> list[dict[str, Any]]:
    chunks = CHUNKER.chunk(text)
    output: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        chunk_text = clean_text(getattr(chunk, "text", str(chunk)))
        token_count = int(
            getattr(chunk, "token_count", 0)
            or len(TOKENIZER.encode(chunk_text, add_special_tokens=False))
        )

        output.append(
            {
                "chunk_id": f"{node.node_id}::chunk-{idx:04d}",
                "chunk_kind": "admonition",
                "source_node_id": node.node_id,
                "parent_id": node.parent_id,
                "page_id": page_id,
                "node_kind": node.node_kind,
                "level": int(node.fence_depth),
                "heading_text": node.title or "",
                "section_url": node.section_url,
                "chunk_index": idx,
                "chunk_text": chunk_text,
                "token_count": token_count,
                "kind": node.kind,
                "title": node.title,
            }
        )

    return output


def build_heading_chunks(
    nodes: Iterable[FlatNode],
    page_id: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for node in nodes:
        text = heading_text(node)
        if not text:
            continue
        output.extend(chunk_heading_node(node, page_id, "heading", text))

    return output


def build_admonition_chunks(
    nodes: Iterable[AdmonitionNode],
    page_id: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for node in nodes:
        text = admonition_text(node)
        if not text:
            continue
        output.extend(chunk_admonition_node(node, page_id, text))

    return output


def build_h2_subtree_chunks_from_node(
    node: Node,
    page_id: str,
) -> list[dict[str, Any]]:
    """
    Strategy:
    - keep H1 as its own short chunk
    - aggregate H2 with all descendant H3/H4/... heading content
    - keep admonitions separate for now
    """
    output: list[dict[str, Any]] = []

    if node.level == 2:
        text = collect_subtree_text(node)
        if text:
            output.extend(chunk_heading_node(node, page_id, "heading", text))
        return output

    for child in node.children:
        output.extend(build_h2_subtree_chunks_from_node(child, page_id))

    return output


def build_h2_subtree_chunks(page_tree: PageTree) -> list[dict[str, Any]]:
    page_id = page_tree.page.page_id
    chunks: list[dict[str, Any]] = []

    for root in page_tree.roots:
        if root.level == 1:
            root_text = collect_node_text(root)
            if root_text:
                chunks.extend(chunk_heading_node(root, page_id, "heading", root_text))

            for child in root.children:
                chunks.extend(build_h2_subtree_chunks_from_node(child, page_id))
        else:
            chunks.extend(build_h2_subtree_chunks_from_node(root, page_id))

    chunks.extend(build_admonition_chunks(page_tree.admonition_nodes, page_id))
    return chunks


def build_chunks_from_page_tree(
    page_tree: PageTree,
    strategy: str = "section",
) -> list[dict[str, Any]]:
    if strategy == "section":
        page_id = page_tree.page.page_id
        chunks: list[dict[str, Any]] = []
        chunks.extend(build_heading_chunks(page_tree.flat_nodes, page_id))
        chunks.extend(build_admonition_chunks(page_tree.admonition_nodes, page_id))
        return chunks

    if strategy == "h2_subtree":
        return build_h2_subtree_chunks(page_tree)

    raise ValueError(f"Unknown chunking strategy: {strategy}")


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is a reusable chunking core. Use a runner script to load page trees and call build_chunks_from_page_tree()."
    )