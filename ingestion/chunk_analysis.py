from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, TypedDict

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment]


PROCESSED_DIR = Path("processed")
HEADINGS_PATH = PROCESSED_DIR / "first_steps_headings_by_id.json"
OUTPUT_PATH = PROCESSED_DIR / "first_steps_h3_h4_stats.json"
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256


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


class HeadingRecord(TypedDict):
    node_id: str
    parent_id: str | None
    level: int
    node_kind: str
    heading_text: str
    anchor_id: str | None
    section_url: str
    section_text: str
    section_inline_code: list[str]
    section_links: list[LinkRef]
    code_refs: list[CodeRef]
    code_blocks: list[CodeBlock]


class HeadingStat(TypedDict):
    node_id: str
    parent_id: str | None
    level: int
    heading_text: str
    token_count: int
    char_count: int


class BundleStat(TypedDict):
    node_id: str
    heading_text: str
    child_count: int
    token_count: int
    char_count: int
    exceeds_limit: bool


class StatsOutput(TypedDict):
    tokenizer_name: str
    max_tokens: int
    total_headings: int
    level_counts: dict[str, int]
    level_3_stats: list[HeadingStat]
    level_4_stats: list[HeadingStat]
    level_3_bundle_stats: list[BundleStat]
    level_4_total_tokens: dict[str, int]
    level_4_over_limit_count: int
    level_4_over_limit_nodes: list[str]


@dataclass(frozen=True)
class TokenizerWrapper:
    tokenizer: Any | None

    def count(self, text: str) -> int:
        if self.tokenizer is None:
            return estimate_tokens(text)
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def estimate_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def load_tokenizer() -> TokenizerWrapper:
    if AutoTokenizer is None:
        return TokenizerWrapper(tokenizer=None)
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    except Exception:
        tokenizer = None
    return TokenizerWrapper(tokenizer=tokenizer)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def node_sort_key(node_id: str) -> tuple[str, int]:
    match = re.search(r"section-(\d+)$", node_id)
    if match:
        return node_id.rsplit("::", 1)[0], int(match.group(1))
    return node_id, 0


def build_children_map(headings_by_id: dict[str, HeadingRecord]) -> dict[str, list[HeadingRecord]]:
    children_map: dict[str, list[HeadingRecord]] = {}
    for node in headings_by_id.values():
        parent_id = node["parent_id"]
        if parent_id is None:
            continue
        children_map.setdefault(parent_id, []).append(node)

    for parent_id in list(children_map.keys()):
        children_map[parent_id].sort(key=lambda n: node_sort_key(n["node_id"]))
    return children_map


def merged_h3_text(node: HeadingRecord, children_map: dict[str, list[HeadingRecord]]) -> str:
    parts: list[str] = []
    if node["heading_text"]:
        parts.append(node["heading_text"])
    if node["section_text"]:
        parts.append(node["section_text"])

    for child in children_map.get(node["node_id"], []):
        if child["node_kind"] != "heading" or child["level"] != 4:
            continue
        if child["heading_text"]:
            parts.append(child["heading_text"])
        if child["section_text"]:
            parts.append(child["section_text"])

    return "\n\n".join(part for part in parts if part.strip())


def heading_stat(node: HeadingRecord, tokenizer: TokenizerWrapper) -> HeadingStat:
    text = "\n\n".join(part for part in [node["heading_text"], node["section_text"]] if part.strip())
    return {
        "node_id": node["node_id"],
        "parent_id": node["parent_id"],
        "level": node["level"],
        "heading_text": node["heading_text"],
        "token_count": tokenizer.count(text),
        "char_count": len(text),
    }


def bundle_stat(node: HeadingRecord, bundle_text: str, tokenizer: TokenizerWrapper, child_count: int) -> BundleStat:
    return {
        "node_id": node["node_id"],
        "heading_text": node["heading_text"],
        "child_count": child_count,
        "token_count": tokenizer.count(bundle_text),
        "char_count": len(bundle_text),
        "exceeds_limit": tokenizer.count(bundle_text) > MAX_TOKENS,
    }


def main() -> None:
    headings_raw = load_json(HEADINGS_PATH)
    headings_by_id: dict[str, HeadingRecord] = {node_id: node for node_id, node in headings_raw.items()}
    tokenizer = load_tokenizer()
    children_map = build_children_map(headings_by_id)

    ordered_nodes = [headings_by_id[node_id] for node_id in sorted(headings_by_id.keys(), key=node_sort_key)]
    level_counts: dict[str, int] = {}
    level_3_stats: list[HeadingStat] = []
    level_4_stats: list[HeadingStat] = []
    level_3_bundle_stats: list[BundleStat] = []
    level_4_total_tokens: dict[str, int] = {}
    level_4_over_limit_nodes: list[str] = []

    for node in ordered_nodes:
        level_key = str(node["level"])
        level_counts[level_key] = level_counts.get(level_key, 0) + 1

        stat = heading_stat(node, tokenizer)
        if node["level"] == 3:
            level_3_stats.append(stat)
        elif node["level"] == 4:
            level_4_stats.append(stat)
            level_4_total_tokens[node["node_id"]] = stat["token_count"]
            if stat["token_count"] > MAX_TOKENS:
                level_4_over_limit_nodes.append(node["node_id"])

    for node in ordered_nodes:
        if node["level"] != 3:
            continue
        child_count = len([child for child in children_map.get(node["node_id"], []) if child["node_kind"] == "heading" and child["level"] == 4])
        bundle = merged_h3_text(node, children_map)
        level_3_bundle_stats.append(bundle_stat(node, bundle, tokenizer, child_count))

    stats: StatsOutput = {
        "tokenizer_name": TOKENIZER_NAME,
        "max_tokens": MAX_TOKENS,
        "total_headings": len(headings_by_id),
        "level_counts": level_counts,
        "level_3_stats": level_3_stats,
        "level_4_stats": level_4_stats,
        "level_3_bundle_stats": level_3_bundle_stats,
        "level_4_total_tokens": level_4_total_tokens,
        "level_4_over_limit_count": len(level_4_over_limit_nodes),
        "level_4_over_limit_nodes": level_4_over_limit_nodes,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Wrote stats to {OUTPUT_PATH}")
    print(f"Total headings: {stats['total_headings']}")
    print(f"Level counts: {stats['level_counts']}")
    print(f"Level 4 over limit: {stats['level_4_over_limit_count']}")
    if level_3_bundle_stats:
        worst = sorted(level_3_bundle_stats, key=lambda item: item["token_count"], reverse=True)[:5]
        print("Top H3 bundles:")
        for item in worst:
            print(
                f"  {item['node_id']} | tokens={item['token_count']} | children={item['child_count']} | exceeds={item['exceeds_limit']}"
            )


if __name__ == "__main__":
    main()
