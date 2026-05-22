from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Mapping

from chonkie import SentenceChunker
from transformers import AutoTokenizer


PROCESSED_DIR = Path("processed")
HEADINGS_PATH = PROCESSED_DIR / "first_steps_headings_by_id.json"
ADMONITIONS_PATH = PROCESSED_DIR / "first_steps_admonitions_by_id.json"
OUTPUT_PATH = PROCESSED_DIR / "first_steps_chunks.json"

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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_page_id(headings_by_id: Mapping[str, dict[str, Any]]) -> str:
    if not headings_by_id:
        return ""
    first = next(iter(headings_by_id.values()))
    return str(first["node_id"]).split("::", 1)[0]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def heading_text(node: Mapping[str, Any]) -> str:
    heading = clean_text(node.get("heading_text") or "")
    body = clean_text(node.get("section_text") or "")
    if heading and body:
        return f"{heading}. {body}"
    return heading or body


def admonition_text(node: Mapping[str, Any]) -> str:
    title = clean_text(node.get("title") or "")
    body = clean_text(node.get("section_text") or "")
    if title and body:
        return f"{title}. {body}"
    return title or body


def chunk_node(node_id: str, node: Mapping[str, Any], page_id: str, chunk_kind: str, text: str) -> list[dict[str, Any]]:
    chunks = CHUNKER.chunk(text)
    output: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        chunk_text = clean_text(getattr(chunk, "text", str(chunk)))
        token_count = int(getattr(chunk, "token_count", 0) or len(TOKENIZER.encode(chunk_text, add_special_tokens=False)))
        output.append(
            {
                "chunk_id": f"{node_id}::chunk-{idx:04d}",
                "chunk_kind": chunk_kind,
                "source_node_id": node_id,
                "parent_id": node.get("parent_id"),
                "page_id": page_id,
                "node_kind": node.get("node_kind", chunk_kind),
                "level": int(node.get("level", 0)),
                "heading_text": node.get("heading_text", node.get("title") or ""),
                "section_url": node.get("section_url", ""),
                "chunk_index": idx,
                "chunk_text": chunk_text,
                "token_count": token_count,
                "kind": node.get("kind"),
                "title": node.get("title"),
            }
        )

    return output


def build_chunks(nodes: Mapping[str, Mapping[str, Any]], page_id: str, chunk_kind: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for node_id, node in nodes.items():
        text = heading_text(node) if chunk_kind == "heading" else admonition_text(node)
        if not text:
            continue
        output.extend(chunk_node(node_id, node, page_id, chunk_kind, text))

    return output


def main() -> None:
    headings_raw = load_json(HEADINGS_PATH)
    admonitions_raw = load_json(ADMONITIONS_PATH)

    headings_by_id: dict[str, dict[str, Any]] = {node_id: node for node_id, node in headings_raw.items()}
    admonitions_by_id: dict[str, dict[str, Any]] = {node_id: node for node_id, node in admonitions_raw.items()}

    page_id = infer_page_id(headings_by_id)

    chunks: list[dict[str, Any]] = []
    chunks.extend(build_chunks(headings_by_id, page_id, "heading"))
    chunks.extend(build_chunks(admonitions_by_id, page_id, "admonition"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH}")
    print()
    print("Sample chunks:")
    for sample in chunks[:5]:
        print(f"- {sample['chunk_id']} | {sample['chunk_kind']} | tokens={sample['token_count']}")
        print(f"  {sample['chunk_text'][:220]}")
        print()


if __name__ == "__main__":
    main()
