from __future__ import annotations

from typing import Any, Iterable
import re

from chonkie import SentenceChunker
from transformers import AutoTokenizer

from parse_markdown import AdmonitionNode, FlatNode, PageTree


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 230
CHUNK_OVERLAP = 24

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


def chunk_heading_node(
    node: FlatNode,
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
                "chunk_kind": "heading",
                "source_node_id": node.node_id,
                "parent_id": node.parent_id,
                "page_id": page_id,
                "node_kind": node.node_kind,
                "level": int(node.level),
                "heading_text": node.heading_text,
                "section_url": node.section_url,
                "chunk_index": idx,
                "chunk_text": chunk_text,
                "token_count": token_count,
                "kind": None,
                "title": None,
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
                # fence_depth is the best available structural signal for nested blocks
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
        output.extend(chunk_heading_node(node, page_id, text))

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


def build_chunks_from_page_tree(page_tree: PageTree) -> list[dict[str, Any]]:
    page_id = page_tree.page.page_id

    chunks: list[dict[str, Any]] = []
    chunks.extend(build_heading_chunks(page_tree.flat_nodes, page_id))
    chunks.extend(build_admonition_chunks(page_tree.admonition_nodes, page_id))
    return chunks


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is a reusable chunking core. Use a runner script to load page trees and call build_chunks_from_page_tree()."
    )