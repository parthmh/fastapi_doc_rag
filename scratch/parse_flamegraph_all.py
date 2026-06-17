import re

def main():
    svg_path = "processed/profile_gil_clean.svg"
    with open(svg_path, "r", encoding="utf-8") as f:
        content = f.read()

    total_samples_match = re.search(r'total_samples="(\d+)"', content)
    total_samples = int(total_samples_match.group(1)) if total_samples_match else 0
    print(f"Total Samples in 1-Worker Profile: {total_samples}")

    titles = re.findall(r'<title>([^<]+)</title>', content)
    print(f"Total frame titles found: {len(titles)}")

    parsed_frames = []
    for title in titles:
        # Regex to parse: "func (file:line) (N samples, P%)"
        match = re.search(r'^(.*?) \(([\d,]+) samples, (.*?)\)$', title)
        if match:
            func_desc = match.group(1).strip()
            samples = int(match.group(2).replace(",", ""))
            pct = match.group(3).strip()
            parsed_frames.append((func_desc, samples, pct))
        else:
            match_simple = re.search(r'^(.*?) \(([\d,]+) samples', title)
            if match_simple:
                func_desc = match_simple.group(1).strip()
                samples = int(match_simple.group(2).replace(",", ""))
                parsed_frames.append((func_desc, samples, "unknown"))

    parsed_frames.sort(key=lambda x: x[1], reverse=True)

    print("\n=== TOP 40 ACTIVE/WAITING FRAMES BY SAMPLE COUNT (1-WORKER) ===")
    for i, (func_desc, samples, pct) in enumerate(parsed_frames[:40]):
        print(f"{i+1:2d}. [{samples:5d} samples / {pct:5s}] {func_desc}")

if __name__ == "__main__":
    main()
