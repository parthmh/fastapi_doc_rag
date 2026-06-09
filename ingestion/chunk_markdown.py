from __future__ import annotations

from typing import Any, Iterable
import re

from chonkie import SentenceChunker
from transformers import AutoTokenizer

from .parse_markdown import AdmonitionNode, FlatNode, Node, PageTree


from functools import lru_cache
from app.config import settings

CHUNK_OVERLAP = 24

SECTION_SEPARATOR = "\n\n"

@lru_cache(maxsize=2)
def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

@lru_cache(maxsize=2)
def get_chunker(model_name: str, chunk_size: int, chunk_overlap: int):
    tokenizer = get_tokenizer(model_name)
    return SentenceChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
    tokenizer = get_tokenizer(settings.dense_model_name)
    chunker = get_chunker(settings.dense_model_name, settings.chunk_size, CHUNK_OVERLAP)
    chunks = chunker.chunk(text)
    output: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        chunk_text = clean_text(getattr(chunk, "text", str(chunk)))
        token_count = int(
            getattr(chunk, "token_count", 0)
            or len(tokenizer.encode(chunk_text, add_special_tokens=False))
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
    tokenizer = get_tokenizer(settings.dense_model_name)
    chunker = get_chunker(settings.dense_model_name, settings.chunk_size, CHUNK_OVERLAP)
    chunks = chunker.chunk(text)
    output: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        chunk_text = clean_text(getattr(chunk, "text", str(chunk)))
        token_count = int(
            getattr(chunk, "token_count", 0)
            or len(tokenizer.encode(chunk_text, add_special_tokens=False))
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


def bypass_chunk_node(
    node: Node | FlatNode | AdmonitionNode,
    page_id: str,
    chunk_kind: str,
    text: str,
) -> list[dict[str, Any]]:
    tokenizer = get_tokenizer(settings.dense_model_name)
    chunk_text = clean_text(text)
    token_count = len(tokenizer.encode(chunk_text, add_special_tokens=False))
    
    if isinstance(node, AdmonitionNode):
        level = int(node.fence_depth)
        heading_text_val = node.title or ""
        kind_val = node.kind
        title_val = node.title
    else:
        level = int(getattr(node, "level", 0))
        heading_text_val = getattr(node, "heading_text", "")
        kind_val = getattr(node, "kind", None)
        title_val = getattr(node, "title", None)
        
    return [
        {
            "chunk_id": f"{node.node_id}::chunk-0000",
            "chunk_kind": chunk_kind,
            "source_node_id": node.node_id,
            "parent_id": node.parent_id,
            "page_id": page_id,
            "node_kind": node.node_kind,
            "level": level,
            "heading_text": heading_text_val,
            "section_url": node.section_url,
            "chunk_index": 0,
            "chunk_text": chunk_text,
            "token_count": token_count,
            "kind": kind_val,
            "title": title_val,
        }
    ]


def build_chunks_from_page_tree(
    page_tree: PageTree,
    strategy: str = "section",
) -> list[dict[str, Any]]:
    if not settings.chunking_enabled:
        page_id = page_tree.page.page_id
        chunks: list[dict[str, Any]] = []
        for node in page_tree.flat_nodes:
            text = heading_text(node)
            if text:
                chunks.extend(bypass_chunk_node(node, page_id, "heading", text))
        for node in page_tree.admonition_nodes:
            text = admonition_text(node)
            if text:
                chunks.extend(bypass_chunk_node(node, page_id, "admonition", text))
        return chunks

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