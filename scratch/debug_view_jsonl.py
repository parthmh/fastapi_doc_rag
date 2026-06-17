# debug_view_jsonl.py
import json
from pathlib import Path

path = Path("processed/first_steps_sections.jsonl")

for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
        continue
    r = json.loads(line)
    print("=" * 80)
    print(f"{i}. {r['anchor_id']}")
    print(f"Level: {r['heading_level']}")
    print(f"Path: {' > '.join(r['heading_path'])}")
    print("\nCANONICAL TEXT\n")
    print(r["canonical_text"])
    print("\nSIGNALS")
    print("inline_code:", r["signals"]["inline_code"])
    print("bold_terms:", r["signals"]["bold_terms"])
    print("links:", r["signals"]["links"])
    print("code_refs:", r["signals"]["code_refs"])
    print()