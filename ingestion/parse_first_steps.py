from pathlib import Path
import re
from typing import Any, cast
import pyromark

CORPUS_ROOT = Path("corpus")
MARKDOWN_PATH = CORPUS_ROOT / "tutorial" / "first-steps.md"

HEADING_WITH_ANCHOR_RE = re.compile(
    r"^(?P<title>.*?)\s*\{\s*#(?P<anchor>[^}]+)\s*\}\s*$"
)


def parse_heading_text(raw_heading_text: str) -> tuple[str, str | None]:
    m = HEADING_WITH_ANCHOR_RE.match(raw_heading_text.strip())
    if m:
        return m.group("title").strip(), m.group("anchor").strip()
    return raw_heading_text.strip(), None


if __name__ == "__main__":
    markdown_text = MARKDOWN_PATH.read_text(encoding="utf-8")

    inside_heading = False
    current_heading_level = None
    heading_parts: list[str] = []
    
    for event in pyromark.events(markdown_text):
        e = cast(dict[str, Any], event)

        if "Start" in e:
            start_value = e["Start"]
            if isinstance(start_value, dict) and "Heading" in start_value:
                inside_heading = True
                current_heading_level = start_value["Heading"]["level"]
                heading_parts = []

        elif inside_heading:
            if "Text" in e:
                heading_parts.append(e["Text"])
            elif "Code" in e:
                heading_parts.append(f"`{e['Code']}`")
            elif "End" in e:
                end_value = e["End"]
                if isinstance(end_value, dict) and "Heading" in end_value:
                    raw_heading = "".join(heading_parts).strip()
                    heading_text, anchor_id = parse_heading_text(raw_heading)

                    print("=" * 60)
                    print("LEVEL: ", current_heading_level)
                    print("RAW:   ", raw_heading)
                    print("TEXT:  ", heading_text)
                    print("ANCHOR:", anchor_id)
            
                    inside_heading = False
                    current_heading_level = None
                    heading_parts = []
    print(pyromark.events(markdown_text))