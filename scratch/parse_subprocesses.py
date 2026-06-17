import re

def main():
    svg_path = "processed/profile_gil_clean.svg"
    with open(svg_path, "r", encoding="utf-8") as f:
        content = f.read()

    total_samples_match = re.search(r'total_samples="(\d+)"', content)
    total_samples = int(total_samples_match.group(1)) if total_samples_match else 0
    print(f"Total Samples in Profile: {total_samples}\n")

    # Find all <title> tags inside <g> elements
    titles = re.findall(r'<title>([^<]+)</title>', content)

    print(f"DEBUG: Found {len(titles)} total titles.")
    print("DEBUG: Showing first 10 titles starting with 'process':")
    debug_count = 0
    for title in titles:
        if title.strip().startswith("process"):
            print(f"  - {title}")
            debug_count += 1
            if debug_count >= 10:
                break

    process_samples = {}
    for title in titles:
        if title.strip().startswith("process"):
            # Match process boxes, handling commas in numbers like "2,859"
            match = re.search(r'^(process \d+:[^(\n]+) \(([\d,]+) samples, (.*?)\)$', title)
            if match:
                proc_name = match.group(1).strip()
                samples_str = match.group(2).replace(",", "")
                samples = int(samples_str)
                pct = match.group(3).strip()
                process_samples[proc_name] = (samples, pct)
            else:
                match_simple = re.search(r'^(process \d+:[^(\n]+) \(([\d,]+) samples', title)
                if match_simple:
                    proc_name = match_simple.group(1).strip()
                    samples_str = match_simple.group(2).replace(",", "")
                    samples = int(samples_str)
                    process_samples[proc_name] = (samples, "unknown")

    print("\n=== SAMPLES / GIL LOCKING BY PROCESS / SUBPROCESS ===")
    sorted_procs = sorted(process_samples.items(), key=lambda x: x[1][0], reverse=True)
    
    total_accounted = 0
    for proc_name, (samples, pct) in sorted_procs:
        print(f"Process: {proc_name}")
        print(f"  Samples:    {samples:,} ({pct})")
        total_accounted += samples
        
    print(f"\nTotal Accounted Process Samples: {total_accounted:,} / {total_samples:,} ({total_accounted/total_samples*100:.2f}%)")

if __name__ == "__main__":
    main()
