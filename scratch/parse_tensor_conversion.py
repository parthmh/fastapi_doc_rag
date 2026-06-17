import re

def main():
    svg_path = "processed/profile_gil_clean.svg"
    with open(svg_path, "r", encoding="utf-8") as f:
        content = f.read()

    titles = re.findall(r'<title>([^<]+)</title>', content)
    
    # Target keywords related to PyTorch C++ to Python conversion
    keywords = ["tolist", "numpy", "tensor", "convert", "array", "cpu", "float"]
    
    print("=== TENSOR / FLOAT / CONVERSION FRAMES ===")
    matches = []
    for title in titles:
        if any(kw in title.lower() for kw in keywords):
            match = re.search(r'^(.*?) \(([\d,]+) samples, (.*?)\)$', title)
            if match:
                func_desc = match.group(1).strip()
                samples = int(match.group(2).replace(",", ""))
                pct = match.group(3).strip()
                matches.append((func_desc, samples, pct))
            else:
                match_simple = re.search(r'^(.*?) \(([\d,]+) samples', title)
                if match_simple:
                    func_desc = match_simple.group(1).strip()
                    samples = int(match_simple.group(2).replace(",", ""))
                    matches.append((func_desc, samples, "unknown"))

    matches.sort(key=lambda x: x[1], reverse=True)
    for i, (func_desc, samples, pct) in enumerate(matches[:40]):
        print(f"{i+1:2d}. [{samples:5d} samples / {pct:5s}] {func_desc}")

if __name__ == "__main__":
    main()
