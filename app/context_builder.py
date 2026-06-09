import re
from pathlib import Path
from typing import Any, List
from app.retriever import RetrievedChunk

def extract_section_by_heading(markdown_text: str, heading_text: str) -> str:
    """
    Extract lines of a markdown file belonging to a specific heading.
    Extracts up to the next heading of equal or higher level.
    """
    lines = markdown_text.splitlines()
    section_lines = []
    in_section = False
    section_level = 0
    heading_clean = heading_text.strip().lower().replace("`", "")

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("#"):
            parts = stripped_line.split(None, 1)
            # Determine heading level from # count
            level = len(parts[0])
            # Remove any trailing anchor brackets like { #anchor } and strip backticks
            raw_h_text = parts[1].strip() if len(parts) > 1 else ""
            h_text_no_anchor = re.sub(r"\{.*?\}", "", raw_h_text).strip()
            h_text = h_text_no_anchor.lower().replace("`", "")

            if in_section:
                # End section when encountering another heading of same or higher level
                if level <= section_level:
                    break
            elif h_text == heading_clean or heading_clean in h_text or h_text in heading_clean:
                in_section = True
                section_level = level

        if in_section:
            section_lines.append(line)

    return "\n".join(section_lines) if section_lines else ""

def resolve_and_inject_code(markdown_section: str, md_file_path: Path, corpus_root: str = "corpus") -> str:
    """
    Finds placeholder patterns like {* path hl[lines] *} in the section text,
    resolves the path of the source code, reads the code content, and replaces
    the placeholder with a formatted markdown code block.
    """
    def replace_ref(match: re.Match) -> str:
        raw_spec = match.group(1).strip()
        parts = raw_spec.split()
        if not parts:
            return match.group(0)
        
        ref_path = parts[0]
        # All code source is located in corpus/docs_src. Map relative URLs under docs_src to this root.
        if "docs_src/" in ref_path:
            clean_subpath = ref_path.split("docs_src/", 1)[-1]
            resolved_path = (Path(corpus_root) / "docs_src" / clean_subpath).resolve()
        else:
            resolved_path = (md_file_path.parent / ref_path).resolve()
        
        if not resolved_path.exists():
            return f"\n*Code reference file not found: {ref_path}*\n"

        try:
            code_content = resolved_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"\n*Error reading code file {ref_path}: {e}*\n"

        # Check for hl[...] highlighting lines specification
        highlight_lines = []
        hl_match = re.search(r"hl\[(.*?)\]", raw_spec)
        if hl_match:
            hl_spec = hl_match.group(1).strip()
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

        # Check for title specification e.g., title="filename.py"
        title = ""
        title_match = re.search(r"title\[\"(.*?)\"\]", raw_spec)
        if title_match:
            title = f" | File: {title_match.group(1)}"

        # Select file suffix as code language indicator
        lang = resolved_path.suffix.lstrip(".")
        if lang == "py":
            lang = "python"

        # Format header showing the file path
        header_comment = f"# File: {resolved_path.name}{title}\n"
        
        # If highlighting is requested, mark selected lines
        if highlight_lines:
            code_lines = code_content.splitlines()
            formatted_lines = []
            for idx, line in enumerate(code_lines, start=1):
                if idx in highlight_lines:
                    formatted_lines.append(f"{line}  # <--- highlighted")
                else:
                    formatted_lines.append(line)
            code_content = "\n".join(formatted_lines)

        return f"\n```{lang}\n{header_comment}{code_content}\n```\n"

    # Match syntax: {* ../../docs_src/path.py hl[1,3:5] *}
    return re.sub(r"\{\*\s*(.*?)\s*\*\}", replace_ref, markdown_section)

import asyncio
from anyio.to_thread import run_sync
from typing import Any

def process_single_chunk(chunk: RetrievedChunk, corpus_root: str) -> str:
    """
    Synchronously processes a single retrieved chunk:
    Loads the source markdown file, extracts the specific heading section,
    resolves and formats code references, and formats the XML block.
    """
    payload = chunk.payload
    page_id = payload.get("page_id", "")
    source_file = payload.get("source_file") or (f"{page_id}.md" if page_id else "")
    heading_text = payload.get("heading_text", "")
    section_url = payload.get("section_url", "")
    chunk_text = payload.get("chunk_text", "")

    # Fallback to raw chunk text if file extraction is unavailable
    final_markdown_content = chunk_text

    if source_file and heading_text:
        md_file_path = Path(corpus_root) / source_file
        if md_file_path.exists():
            try:
                markdown_text = md_file_path.read_text(encoding="utf-8")
                extracted_section = extract_section_by_heading(markdown_text, heading_text)
                if extracted_section:
                    # Inject referenced source code files inline
                    final_markdown_content = resolve_and_inject_code(extracted_section, md_file_path, corpus_root)
            except Exception:
                pass

    return (
        f'<retrieved_document page_id="{page_id}" heading="{heading_text}" url="{section_url}">\n'
        f"{final_markdown_content}\n"
        f"</retrieved_document>"
    )

async def reconstruct_context(retrieved_chunks: list[RetrievedChunk], corpus_root: str = "corpus") -> str:
    """
    Spawns concurrent worker threads to process all retrieved chunks in parallel,
    maximizing file I/O throughput.
    """
    tasks = [
        run_sync(process_single_chunk, chunk, corpus_root)
        for chunk in retrieved_chunks
    ]
    # Execute all chunk formatting and I/O tasks concurrently in background threads
    documents_context = await asyncio.gather(*tasks)
    raw_context = "\n\n".join(documents_context)
    # Collapse 3 or more consecutive newlines down to exactly 2 to prevent token waste
    return re.sub(r"\n{3,}", "\n\n", raw_context).strip()
