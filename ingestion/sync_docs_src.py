from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_ROOT = REPO_ROOT / "corpus" / "tutorial"
LOCAL_DOCS_SRC_ROOT = REPO_ROOT / "corpus" / "docs_src"
EXTERNAL_DOCS_SRC_ROOT = Path.home() / "learn" / "fastapi" / "docs_src"

INCLUDE_RE = re.compile(r"\{\*\s*(.*?)\s*\*\}")
HL_RE = re.compile(r"\s+hl\[[^\]]*\]$")


@dataclass(frozen=True)
class IncludeRef:
    tutorial_file: Path
    raw_path: str
    resolved_external_path: Path
    relative_docs_src_path: Path


def iter_tutorial_markdown_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.md"))


def extract_include_paths(markdown_text: str) -> list[str]:
    paths: list[str] = []

    for match in INCLUDE_RE.finditer(markdown_text):
        inner = match.group(1).strip()

        if not inner:
            continue

        raw_path = inner.split()[0].strip()

        if raw_path:
            paths.append(raw_path.replace("\\", "/"))

    return paths


def resolve_include_path(raw_include_path: str) -> Path:
    """
    Convert include paths like:
        ../../docs_src/background_tasks/tutorial001_py310.py
    into:
        background_tasks/tutorial001_py310.py
    """
    normalized = raw_include_path.replace("\\", "/").strip()
    parts = PurePosixPath(normalized).parts

    if "docs_src" not in parts:
        raise ValueError(f"Include path does not contain docs_src/: {raw_include_path}")

    docs_src_index = parts.index("docs_src")
    relative_parts = parts[docs_src_index + 1 :]

    if not relative_parts:
        raise ValueError(f"Include path points to docs_src root only: {raw_include_path}")

    return Path(*relative_parts)


def collect_required_sources() -> list[IncludeRef]:
    refs: list[IncludeRef] = []
    seen: set[Path] = set()

    for tutorial_file in iter_tutorial_markdown_files(TUTORIAL_ROOT):
        markdown_text = tutorial_file.read_text(encoding="utf-8")
        include_paths = extract_include_paths(markdown_text)

        for raw_path in include_paths:
            relative_docs_src_path = resolve_include_path(raw_path)
            external_path = (EXTERNAL_DOCS_SRC_ROOT / relative_docs_src_path).resolve()

            if external_path in seen:
                continue

            if not external_path.exists():
                raise FileNotFoundError(
                    f"Referenced source file does not exist:\n"
                    f"  tutorial file: {tutorial_file}\n"
                    f"  include path:  {raw_path}\n"
                    f"  resolved to:   {external_path}"
                )

            seen.add(external_path)
            refs.append(
                IncludeRef(
                    tutorial_file=tutorial_file,
                    raw_path=raw_path,
                    resolved_external_path=external_path,
                    relative_docs_src_path=relative_docs_src_path,
                )
            )

    return refs


def copy_required_sources(refs: list[IncludeRef]) -> None:
    LOCAL_DOCS_SRC_ROOT.mkdir(parents=True, exist_ok=True)

    for ref in refs:
        dest = LOCAL_DOCS_SRC_ROOT / ref.relative_docs_src_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ref.resolved_external_path, dest)


def write_manifest(refs: list[IncludeRef]) -> None:
    manifest_path = REPO_ROOT / "processed" / "docs_src_manifest.txt"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for ref in refs:
        lines.append(
            f"{ref.tutorial_file.relative_to(REPO_ROOT)} -> "
            f"{ref.relative_docs_src_path.as_posix()}"
        )

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


def main() -> None:
    if not TUTORIAL_ROOT.exists():
        raise FileNotFoundError(f"Tutorial root not found: {TUTORIAL_ROOT}")

    if not EXTERNAL_DOCS_SRC_ROOT.exists():
        raise FileNotFoundError(
            f"External docs_src root not found: {EXTERNAL_DOCS_SRC_ROOT}"
        )

    refs = collect_required_sources()

    print(f"Found {len(refs)} unique referenced source files.")
    for ref in refs:
        print(f"- {ref.relative_docs_src_path.as_posix()}")

    copy_required_sources(refs)
    write_manifest(refs)

    print()
    print(f"Copied required source files into: {LOCAL_DOCS_SRC_ROOT}")


if __name__ == "__main__":
    main()