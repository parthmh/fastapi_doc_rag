from pathlib import Path
import json
from collections import Counter


CHUNKS_PATH = Path("processed") / "first_steps_chunks.json"


with CHUNKS_PATH.open("r", encoding="utf-8") as f:
    chunks = json.load(f)


chunk_kind_counter = Counter()
node_kind_counter = Counter()

for chunk in chunks:
    chunk_kind_counter[str(chunk.get("chunk_kind"))] += 1
    node_kind_counter[str(chunk.get("node_kind"))] += 1


print()
print("CHUNK KIND COUNTS")
print("=" * 80)

for key, value in chunk_kind_counter.items():
    print(f"{key}: {value}")


print()
print("NODE KIND COUNTS")
print("=" * 80)

for key, value in node_kind_counter.items():
    print(f"{key}: {value}")