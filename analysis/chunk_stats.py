from __future__ import annotations

from collections import Counter
from collections import defaultdict
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEBUG_CHUNK_ROOT = Path("processed") / "debug_chunks"
OUTPUT_DIR = Path("processed") / "analysis"


def iter_chunk_files(root: Path):
    if not root.exists():
        return
    yield from sorted(root.rglob("chunks.json"))


def load_chunks() -> list[dict]:
    chunks: list[dict] = []

    for chunk_file in iter_chunk_files(DEBUG_CHUNK_ROOT):
        with chunk_file.open("r", encoding="utf-8") as f:
            chunks.extend(json.load(f))

    return chunks


def build_dataframe(chunks: list[dict]) -> pd.DataFrame:
    rows = []

    for chunk in chunks:
        rows.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "page_id": chunk.get("page_id"),
                "chunk_kind": chunk.get("chunk_kind"),
                "node_kind": chunk.get("node_kind"),
                "token_count": chunk.get("token_count", 0),
                "heading_text": chunk.get("heading_text"),
                "source_node_id": chunk.get("source_node_id"),
                "chunk_index": chunk.get("chunk_index", 0),
                "level": chunk.get("level", 0),
                "parent_id": chunk.get("parent_id"),
                "kind": chunk.get("kind"),
                "title": chunk.get("title"),
            }
        )

    return pd.DataFrame(rows)


def build_section_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    section_df = (
        df.groupby("source_node_id", as_index=False)
        .agg(
            page_id=("page_id", "first"),
            chunk_kind=("chunk_kind", "first"),
            node_kind=("node_kind", "first"),
            level=("level", "first"),
            parent_id=("parent_id", "first"),
            heading_text=("heading_text", "first"),
            kind=("kind", "first"),
            title=("title", "first"),
            total_tokens=("token_count", "sum"),
            max_chunk_tokens=("token_count", "max"),
            min_chunk_tokens=("token_count", "min"),
            mean_chunk_tokens=("token_count", "mean"),
            chunk_count=("chunk_index", "count"),
        )
    )

    return section_df


def build_section_tree_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per logical section, keeping the tree structure.
    """
    section_df = (
        df.groupby("source_node_id", as_index=False)
        .agg(
            page_id=("page_id", "first"),
            chunk_kind=("chunk_kind", "first"),
            node_kind=("node_kind", "first"),
            level=("level", "first"),
            parent_id=("parent_id", "first"),
            heading_text=("heading_text", "first"),
            kind=("kind", "first"),
            title=("title", "first"),
            own_tokens=("token_count", "sum"),
            chunk_count=("chunk_index", "count"),
        )
    )

    return section_df


def compute_subtree_token_totals(section_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute true subtree token totals:
    own section tokens + all descendant section tokens.
    """
    section_df = section_df.copy()

    own_tokens_by_id = {
        row.source_node_id: int(row.own_tokens)
        for row in section_df.itertuples(index=False)
    }

    parent_by_id = {
        row.source_node_id: row.parent_id
        for row in section_df.itertuples(index=False)
    }

    children_by_parent: dict[str | None, list[str]] = defaultdict(list)
    for row in section_df.itertuples(index=False):
        children_by_parent[row.parent_id].append(row.source_node_id)

    subtree_cache: dict[str, int] = {}

    def subtree_total(node_id: str) -> int:
        if node_id in subtree_cache:
            return subtree_cache[node_id]

        total = own_tokens_by_id.get(node_id, 0)
        for child_id in children_by_parent.get(node_id, []):
            total += subtree_total(child_id)

        subtree_cache[node_id] = total
        return total

    section_df["subtree_tokens"] = section_df["source_node_id"].map(subtree_total)

    return section_df

def print_summary(df: pd.DataFrame) -> None:
    tokens = df["token_count"]

    print()
    print("=" * 80)
    print("CHUNK TOKEN STATISTICS")
    print("=" * 80)

    print(f"count   : {len(tokens)}")
    print(f"min     : {tokens.min()}")
    print(f"max     : {tokens.max()}")
    print(f"mean    : {tokens.mean():.2f}")
    print(f"median  : {tokens.median():.2f}")
    print(f"std     : {tokens.std():.2f}")

    print()
    print("PERCENTILES")
    print("-" * 80)

    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"p{p:<2}    : {np.percentile(tokens, p):.2f}")

    print()
    print("CHUNK TYPES")
    print("-" * 80)
    print(df["chunk_kind"].value_counts())

    print()
    print("NODE TYPES")
    print("-" * 80)
    print(df["node_kind"].value_counts())


def print_fragmentation_stats(df: pd.DataFrame) -> None:
    grouped = df.groupby("source_node_id")
    chunk_counts = grouped.size()

    total_sections = len(chunk_counts)
    total_chunks = len(df)

    single_chunk_sections = int((chunk_counts == 1).sum())
    two_chunk_sections = int((chunk_counts == 2).sum())
    three_plus_sections = int((chunk_counts >= 3).sum())

    compression_ratio = total_chunks / total_sections if total_sections else 0.0

    print()
    print("=" * 80)
    print("SECTION FRAGMENTATION")
    print("=" * 80)

    print(f"total sections           : {total_sections}")
    print(f"total chunks             : {total_chunks}")
    print(f"compression ratio        : {compression_ratio:.2f}")

    print()
    print(f"1 chunk sections         : {single_chunk_sections} ({single_chunk_sections / total_sections:.2%})")
    print(f"2 chunk sections         : {two_chunk_sections} ({two_chunk_sections / total_sections:.2%})")
    print(f"3+ chunk sections        : {three_plus_sections} ({three_plus_sections / total_sections:.2%})")

    print()
    print("CHUNKS PER SECTION")
    print("-" * 80)

    distribution = Counter(chunk_counts.tolist())
    for count in sorted(distribution):
        freq = distribution[count]
        pct = freq / total_sections
        print(f"{count:>2} chunks : {freq:>4} sections ({pct:.2%})")


def print_overflow_stats(df: pd.DataFrame) -> None:
    overflow_df = df[df["chunk_index"] > 0]

    print()
    print("=" * 80)
    print("OVERFLOW CHUNK ANALYSIS")
    print("=" * 80)

    if overflow_df.empty:
        print("No overflow chunks found.")
        return

    tokens = overflow_df["token_count"]

    print(f"overflow chunk count     : {len(overflow_df)}")
    print(f"overflow ratio           : {len(overflow_df) / len(df):.2%}")

    print()
    print(f"min                      : {tokens.min()}")
    print(f"max                      : {tokens.max()}")
    print(f"mean                     : {tokens.mean():.2f}")
    print(f"median                   : {tokens.median():.2f}")
    print(f"std                      : {tokens.std():.2f}")

    print()
    print("OVERFLOW PERCENTILES")
    print("-" * 80)

    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"p{p:<2}                    : {np.percentile(tokens, p):.2f}")

    tiny_chunks = int((tokens < 40).sum())

    print()
    print(f"overflow chunks < 40 tok : {tiny_chunks} ({tiny_chunks / len(overflow_df):.2%})")


def print_section_level_stats(df: pd.DataFrame) -> None:
    section_df = build_section_dataframe(df)
    tokens = section_df["total_tokens"]

    print()
    print("=" * 80)
    print("SECTION-LEVEL TOKEN STATISTICS")
    print("=" * 80)

    print(f"total sections           : {len(section_df)}")
    print(f"min                      : {tokens.min()}")
    print(f"max                      : {tokens.max()}")
    print(f"mean                     : {tokens.mean():.2f}")
    print(f"median                   : {tokens.median():.2f}")
    print(f"std                      : {tokens.std():.2f}")

    print()
    print("SECTION TOKEN PERCENTILES")
    print("-" * 80)

    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"p{p:<2}                    : {np.percentile(tokens, p):.2f}")

    print()
    print("SECTION KIND DISTRIBUTION")
    print("-" * 80)
    print(section_df["node_kind"].value_counts())

    print()
    print("HEADING LEVEL DISTRIBUTION")
    print("-" * 80)
    level_counts = section_df["level"].value_counts().sort_index()
    for level, count in level_counts.items():
        print(f"H{int(level)}: {count}")

    print()
    print("SECTION CHUNK COUNT DISTRIBUTION")
    print("-" * 80)
    chunk_dist = section_df["chunk_count"].value_counts().sort_index()
    for count, freq in chunk_dist.items():
        pct = freq / len(section_df)
        print(f"{count} chunks : {freq} sections ({pct:.2%})")


def print_subheading_distribution(section_df: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("SUBHEADING DISTRIBUTION")
    print("=" * 80)

    child_df = section_df[section_df["parent_id"].notna()].copy()

    if child_df.empty:
        print("No subheadings found.")
        return

    parent_counts = (
        child_df.groupby(["page_id", "parent_id"])
        .size()
        .reset_index(name="child_count")
    )

    parent_lookup = section_df[
        ["source_node_id", "page_id", "level", "heading_text"]
    ].rename(
        columns={
            "source_node_id": "parent_id",
            "level": "parent_level",
            "heading_text": "parent_heading",
        }
    )

    merged = parent_counts.merge(
        parent_lookup,
        on=["page_id", "parent_id"],
        how="left",
    )

    print()
    print("DIRECT CHILD COUNT BY PARENT LEVEL")
    print("-" * 80)
    for parent_level in sorted(merged["parent_level"].dropna().unique()):
        subset = merged[merged["parent_level"] == parent_level]
        if subset.empty:
            continue

        print(f"H{int(parent_level)} parents")
        print(subset["child_count"].value_counts().sort_index())
        print()


def print_h1_h2_distribution(section_df: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("H1 -> H2 DISTRIBUTION")
    print("=" * 80)

    h2_df = section_df[section_df["level"] == 2].copy()
    if h2_df.empty:
        print("No H2 sections found.")
        return

    h1_counts = (
        h2_df.groupby(["page_id", "parent_id"])
        .size()
        .reset_index(name="h2_count")
    )

    print(h1_counts["h2_count"].value_counts().sort_index())

def print_subtree_level_stats(df: pd.DataFrame) -> None:
    section_df = build_section_tree_dataframe(df)
    section_df = compute_subtree_token_totals(section_df)

    print()
    print("=" * 80)
    print("SUBTREE TOKEN STATISTICS BY HEADING LEVEL")
    print("=" * 80)

    for level in sorted(section_df["level"].dropna().unique()):
        subset = section_df[section_df["level"] == level]
        tokens = subset["subtree_tokens"]

        print()
        print(f"H{int(level)}")
        print("-" * 80)
        print(f"count    : {len(subset)}")
        print(f"min      : {tokens.min()}")
        print(f"max      : {tokens.max()}")
        print(f"mean     : {tokens.mean():.2f}")
        print(f"median   : {tokens.median():.2f}")
        print(f"std      : {tokens.std():.2f}")

        print("percentiles")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"p{p:<2}     : {np.percentile(tokens, p):.2f}")

def plot_histogram(df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "chunk_token_histogram.png"

    plt.figure(figsize=(12, 6))
    plt.hist(df["token_count"], bins=50)
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.title("Chunk Token Distribution")
    plt.tight_layout()
    plt.savefig(output_path)

    return output_path


def plot_cdf(df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "chunk_token_cdf.png"

    sorted_tokens = np.sort(df["token_count"])
    y = np.arange(len(sorted_tokens)) / float(len(sorted_tokens))

    plt.figure(figsize=(12, 6))
    plt.plot(sorted_tokens, y)
    plt.xlabel("Token Count")
    plt.ylabel("CDF")
    plt.title("Chunk Token Distribution CDF")
    plt.tight_layout()
    plt.savefig(output_path)

    return output_path


def plot_chunks_per_section(df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "chunks_per_section.png"

    grouped = df.groupby("source_node_id").size()
    distribution = grouped.value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(
        distribution.index.astype(str),
        distribution.values,
    )
    plt.xlabel("Chunks Per Section")
    plt.ylabel("Section Count")
    plt.title("Section Fragmentation Distribution")
    plt.tight_layout()
    plt.savefig(output_path)

    return output_path


def plot_overflow_histogram(df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "overflow_chunk_histogram.png"

    overflow_df = df[df["chunk_index"] > 0]
    if overflow_df.empty:
        return output_path

    plt.figure(figsize=(12, 6))
    plt.hist(overflow_df["token_count"], bins=40)
    plt.xlabel("Overflow Chunk Token Count")
    plt.ylabel("Frequency")
    plt.title("Overflow Chunk Token Distribution")
    plt.tight_layout()
    plt.savefig(output_path)

    return output_path


def plot_section_size_vs_chunks(df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "section_size_vs_chunks.png"

    grouped = df.groupby("source_node_id")

    section_tokens = grouped["token_count"].sum()
    chunk_counts = grouped.size()

    plt.figure(figsize=(10, 6))
    plt.scatter(section_tokens, chunk_counts)
    plt.xlabel("Total Section Tokens")
    plt.ylabel("Chunks Produced")
    plt.title("Section Size vs Chunk Count")
    plt.tight_layout()
    plt.savefig(output_path)

    return output_path


def plot_section_token_distribution(df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "section_token_distribution.png"

    grouped = df.groupby("source_node_id")
    totals = grouped["token_count"].sum()

    plt.figure(figsize=(12, 6))
    plt.hist(totals, bins=50)
    plt.xlabel("Total Section Tokens")
    plt.ylabel("Frequency")
    plt.title("Section-Level Token Distribution")
    plt.tight_layout()
    plt.savefig(output_path)

    return output_path


def plot_h1_h2_distribution(section_df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "h1_h2_distribution.png"

    h2_df = section_df[section_df["level"] == 2].copy()
    if h2_df.empty:
        return output_path

    h1_counts = (
        h2_df.groupby(["page_id", "parent_id"])
        .size()
        .reset_index(name="h2_count")
    )

    dist = h1_counts["h2_count"].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(
        dist.index.astype(str),
        dist.values,
    )
    plt.xlabel("H2s per H1")
    plt.ylabel("H1 Count")
    plt.title("H1 → H2 Distribution")
    plt.tight_layout()
    plt.savefig(output_path)

    return output_path

def plot_subtree_token_distribution_by_level(df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "subtree_token_distribution_by_level.png"

    section_df = build_section_tree_dataframe(df)
    section_df = compute_subtree_token_totals(section_df)

    plt.figure(figsize=(12, 6))

    for level in sorted(section_df["level"].dropna().unique()):
        subset = section_df[section_df["level"] == level]
        if subset.empty:
            continue
        plt.hist(
            subset["subtree_tokens"],
            bins=40,
            alpha=0.5,
            label=f"H{int(level)}",
        )

    plt.xlabel("Subtree Token Count")
    plt.ylabel("Frequency")
    plt.title("Subtree Token Distribution by Heading Level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)

    return output_path


def main() -> None:
    chunks = load_chunks()

    if not chunks:
        print("No chunks found.")
        return

    df = build_dataframe(chunks)
    section_df = build_section_dataframe(df)

    print_summary(df)
    print_fragmentation_stats(df)
    print_overflow_stats(df)
    print_section_level_stats(df)
    print_subheading_distribution(section_df)
    print_h1_h2_distribution(section_df)
    print_subtree_level_stats(df)



    histogram_path = plot_histogram(df)
    cdf_path = plot_cdf(df)
    chunks_per_section_path = plot_chunks_per_section(df)
    overflow_histogram_path = plot_overflow_histogram(df)
    section_scatter_path = plot_section_size_vs_chunks(df)
    section_distribution_path = plot_section_token_distribution(df)
    h1_h2_plot_path = plot_h1_h2_distribution(section_df)
    subtree_plot_path = plot_subtree_token_distribution_by_level(df)

    print()
    print("=" * 80)
    print("OUTPUTS")
    print("=" * 80)
    print(histogram_path)
    print(cdf_path)
    print(chunks_per_section_path)
    print(overflow_histogram_path)
    print(section_scatter_path)
    print(section_distribution_path)
    print(h1_h2_plot_path)
    print(subtree_plot_path)

if __name__ == "__main__":
    main()