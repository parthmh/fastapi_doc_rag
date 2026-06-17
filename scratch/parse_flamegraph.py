import re

def main():
    svg_path = "processed/profile_gil_clean.svg"
    with open(svg_path, "r", encoding="utf-8") as f:
        content = f.read()

    total_samples_match = re.search(r'total_samples="(\d+)"', content)
    total_samples = int(total_samples_match.group(1)) if total_samples_match else 0
    print(f"Total Samples in Profile: {total_samples}")

    titles = re.findall(r'<title>([^<]+)</title>', content)
    
    # Let's search for specific modules of interest
    keywords = ["ingest_worker", "process_ingest_batch", "sentence_transformers", "torch", "onnx", "embed", "forward"]
    
    print("\n=== MATCHING FRAMES FOR TARGET KEYWORDS ===")
    matches = []
    for title in titles:
        # Check if any keyword matches
        if any(kw in title.lower() for kw in keywords):
            match = re.search(r'^(.*?) \((\d+) samples, (.*?)\)$', title)
            if match:
                func_desc = match.group(1).strip()
                samples = int(match.group(2))
                pct = match.group(3).strip()
                matches.append((func_desc, samples, pct))
            else:
                match_simple = re.search(r'^(.*?) \((\d+) samples', title)
                if match_simple:
                    func_desc = match_simple.group(1).strip()
                    samples = int(match_simple.group(2))
                    matches.append((func_desc, samples, "unknown"))
                    
    matches.sort(key=lambda x: x[1], reverse=True)
    for i, (func_desc, samples, pct) in enumerate(matches[:50]):
        print(f"{i+1:2d}. [{samples:5d} samples / {pct:5s}] {func_desc}")

if __name__ == "__main__":
    main()
