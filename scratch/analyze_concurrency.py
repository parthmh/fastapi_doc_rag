import re
from datetime import datetime
from collections import Counter
import os

LOG_PATH = "processed/concurrency_proof.log"

def analyze():
    if not os.path.exists(LOG_PATH):
        print(f"Error: Log file not found at {LOG_PATH}")
        return

    print(f"Analyzing {LOG_PATH} for concurrency statistics...")
    
    timestamps = []
    latencies = []
    queue_sizes = []
    
    # Example line:
    # [2026-06-11T07:13:05.125678] [API Ingest] Accepted batch of 1 items | Queue Size: 2 | Enqueuing Latency: 0.055ms
    pattern = re.compile(r"^\[(.*?)\] \[API Ingest\] Accepted batch of (\d+) items \| Queue Size: (\d+) \| Enqueuing Latency: ([\d.]+)ms")
    
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                ts_str = match.group(1)
                try:
                    # Parse timestamp with microsecond precision
                    ts = datetime.fromisoformat(ts_str)
                    timestamps.append(ts)
                    queue_sizes.append(int(match.group(3)))
                    latencies.append(float(match.group(4)))
                except Exception:
                    pass

    if not timestamps:
        print("No valid logs found in the file.")
        return

    total_requests = len(timestamps)
    start_time = min(timestamps)
    end_time = max(timestamps)
    duration_sec = (end_time - start_time).total_seconds()
    
    # Calculate average throughput
    throughput = total_requests / duration_sec if duration_sec > 0 else 0
    
    # Group by millisecond to check overlapping concurrent requests
    # e.g., "2026-06-11T07:13:05.125"
    ms_buckets = Counter()
    for ts in timestamps:
        ms_key = ts.strftime("%Y-%m-%dT%H:%M:%S") + f".{ts.microsecond // 1000:03d}"
        ms_buckets[ms_key] += 1
        
    # Analyze overlaps
    max_ms_concurrency = max(ms_buckets.values())
    avg_ms_concurrency = sum(ms_buckets.values()) / len(ms_buckets)
    
    # Find number of milliseconds with multiple requests (concurrency > 1)
    concurrent_ms_count = sum(1 for count in ms_buckets.values() if count > 1)
    pct_concurrent_ms = (concurrent_ms_count / len(ms_buckets)) * 100
    
    print("\n================ Concurrency Analysis Results ================")
    print(f"Total Requests Analyzed : {total_requests:,}")
    print(f"Bombardment Start Time   : {start_time.isoformat()}")
    print(f"Bombardment End Time     : {end_time.isoformat()}")
    print(f"Total Bombard Duration   : {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")
    print(f"API Ingestion Rate       : {throughput:.2f} requests/sec")
    print(f"Average Enqueue Latency  : {sum(latencies)/len(latencies):.3f} ms")
    print(f"Max Queue Backlog Size   : {max(queue_sizes):,}")
    print("--------------------------------------------------------------")
    print(f"Max Requests in 1ms      : {max_ms_concurrency} concurrent requests")
    print(f"Avg Requests per active ms: {avg_ms_concurrency:.2f} requests")
    print(f"Overlapping Active ms    : {concurrent_ms_count:,} out of {len(ms_buckets):,} ms ({pct_concurrent_ms:.2f}%)")
    print("==============================================================")

if __name__ == "__main__":
    analyze()
