import subprocess
import time
import sys
import os
import csv
import httpx

QDRANT_URL = "http://localhost:6333/collections/fastapi_doc_ingest_minilm"
HEALTH_URL = "http://localhost:8000/health"
CSV_PREFIX = "scratch/locust_stats"

def run_cmd(args):
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running command {args}: {res.stderr}")
    return res.stdout, res.returncode

def wait_for_backend():
    print("Waiting for backend to boot up...")
    for _ in range(60):
        try:
            resp = httpx.get(HEALTH_URL, timeout=1.0)
            if resp.status_code == 200:
                print("Backend is healthy and running.")
                return True
        except Exception:
            pass
        time.sleep(1.0)
    print("Error: Backend did not start in time.")
    return False

def set_batch_size(batch_size):
    import re
    print(f"Updating INGEST_BATCH_SIZE in docker-compose.yml to {batch_size}...")
    with open("docker-compose.yml", "r") as f:
        content = f.read()
    content_new = re.sub(r"INGEST_BATCH_SIZE=\d+", f"INGEST_BATCH_SIZE={batch_size}", content)
    with open("docker-compose.yml", "w") as f:
        f.write(content_new)

def get_qdrant_points_count():
    try:
        resp = httpx.get(QDRANT_URL, timeout=2.0)
        if resp.status_code == 200:
            return resp.json().get("result", {}).get("points_count", 0)
    except Exception:
        pass
    return 0

def clean_csv_files():
    for ext in ["_stats.csv", "_stats_history.csv", "_failures.csv", "_exceptions.csv"]:
        path = CSV_PREFIX + ext
        if os.path.exists(path):
            os.remove(path)

def parse_locust_results():
    stats_path = CSV_PREFIX + "_stats.csv"
    if not os.path.exists(stats_path):
        print(f"Error: Locust stats file not found at {stats_path}")
        return None
        
    results = {}
    with open(stats_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # We are interested in the row with Name = 'Aggregated' or Name = '/api/v1/ingest'
            if row.get("Name") == "Aggregated":
                results["total_requests"] = int(row.get("Request Count", 0))
                results["total_failures"] = int(row.get("Failure Count", 0))
                results["avg_latency"] = float(row.get("Average Response Time", 0.0))
                results["min_latency"] = float(row.get("Min Response Time", 0.0))
                results["max_latency"] = float(row.get("Max Response Time", 0.0))
                results["median_latency"] = float(row.get("Median Response Time", 0.0))
                results["throughput"] = float(row.get("Requests/s", 0.0))
                results["p90"] = float(row.get("90%", 0.0))
                results["p95"] = float(row.get("95%", 0.0))
                results["p99"] = float(row.get("99%", 0.0))
    return results

def main():
    clean_csv_files()
    
    # Reset concurrency proof log
    log_path = "processed/concurrency_proof.log"
    if os.path.exists(log_path):
        print("Removing existing concurrency proof log...")
        os.remove(log_path)
    
    # 1. Reset collection
    print("Deleting Qdrant collection...")
    run_cmd(["curl", "-X", "DELETE", QDRANT_URL])
    
    # 2. Update config to 1-by-1
    set_batch_size(1)
    
    # 3. Recreate and restart backend container
    print("Recreating and restarting backend container...")
    run_cmd(["docker", "compose", "up", "--build", "-d", "backend"])
    run_cmd(["docker", "compose", "restart", "backend"])
    if not wait_for_backend():
        sys.exit(1)
    time.sleep(2.0)
    
    # 3. Run Locust test in Distributed Mode (1 Master + 8 Workers)
    print("Launching Locust in Distributed Mode (60 seconds, 6000 users, 2000 spawn rate, 8 parallel workers pinned to cores 8-15)...")
    workers = []
    try:
        # Spawn 8 workers in the background, pinned to cores 8-15
        for i in range(8):
            proc = subprocess.Popen([
                "taskset", "-c", "8-15",
                ".venv/bin/locust",
                "-f", "tests/locust_single_chunk.py",
                "--worker"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            workers.append(proc)
            
        time.sleep(1.0) # Allow workers to register
        
        # Run master process, pinned to cores 8-15
        stdout, code = run_cmd([
            "taskset", "-c", "8-15",
            ".venv/bin/locust", 
            "-f", "tests/locust_single_chunk.py", 
            "--headless", 
            "--master",
            "--expect-workers", "8",
            "-u", "6000", 
            "-r", "2000", 
            "--run-time", "60s", 
            "--host", "http://localhost:8000",
            "--csv", CSV_PREFIX
        ])
    finally:
        # Clean up background worker processes
        print("Cleaning up Locust worker processes...")
        for proc in workers:
            proc.terminate()
            proc.wait()
            
    print("\nLocust process finished.")
    
    # Save docker logs to processed/docker_backend_logs.log
    print("Saving docker logs to processed/docker_backend_logs.log...")
    try:
        os.makedirs("processed", exist_ok=True)
        docker_logs, _ = run_cmd(["docker", "logs", "rag_backend"])
        with open("processed/docker_backend_logs.log", "w", encoding="utf-8") as f:
            f.write(docker_logs)
        print("Successfully saved docker logs.")
    except Exception as e:
        print(f"Error saving docker logs: {e}")
    
    # 4. Parse CSV Results
    results = parse_locust_results()
    if not results:
        print("Failed to parse Locust CSV statistics.")
        sys.exit(1)
        
    total_requests = results["total_requests"]
    
    print(f"\n=== Locust Stress Test Results ===")
    print(f"Total Requests Sent: {total_requests:,}")
    print(f"Total Failures:      {results['total_failures']:,} ({results['total_failures']/total_requests*100:.2f}% Error Rate)")
    print(f"Throughput:          {results['throughput']:.2f} reqs/sec")
    print(f"Latency:")
    print(f"  Min:    {results['min_latency']:.1f} ms")
    print(f"  Median: {results['median_latency']:.1f} ms")
    print(f"  Avg:    {results['avg_latency']:.1f} ms")
    print(f"  P90:    {results['p90']:.1f} ms")
    print(f"  P95:    {results['p95']:.1f} ms")
    print(f"  P99:    {results['p99']:.1f} ms")
    print(f"  Max:    {results['max_latency']:.1f} ms")
    
    # 5. Wait for background queue to drain completely by monitoring container logs activity
    print("\nMonitoring background worker logs to detect queue completion...")
    start_time = time.perf_counter()
    import re
    from datetime import datetime
    
    while True:
        current_points = get_qdrant_points_count()
        elapsed = time.perf_counter() - start_time
        
        # Get docker logs with timestamps to check worker activity
        logs, _ = run_cmd(["docker", "logs", "-t", "rag_backend"])
        log_lines = logs.strip().split("\n")
        
        # Find the last write timing entry
        last_io_time = None
        for line in reversed(log_lines):
            match = re.search(r"^([^\s]+)\s+io:\s+\d+\s+ms", line)
            if match:
                try:
                    ts_str = match.group(1)
                    if "." in ts_str:
                        base, rest = ts_str.split(".", 1)
                        rest_clean = re.sub(r"[^\d]", "", rest)[:6]
                        ts_str = f"{base}.{rest_clean}"
                    else:
                        ts_str = ts_str.rstrip("Z")
                    last_io_time = datetime.fromisoformat(ts_str)
                    break
                except Exception:
                    pass
        
        # Calculate time since last background write activity
        idle_time = 0.0
        if last_io_time:
            idle_time = (datetime.utcnow() - last_io_time).total_seconds()
            
        print(f"[{elapsed:.1f}s] Qdrant Points: {current_points:,} | Last Write Activity: {idle_time:.1f}s ago")
        
        # Keep updating processed/docker_backend_logs.log with the latest complete logs
        try:
            docker_logs, _ = run_cmd(["docker", "logs", "rag_backend"])
            with open("processed/docker_backend_logs.log", "w", encoding="utf-8") as f:
                f.write(docker_logs)
        except Exception:
            pass
        
        # If the last write was more than 15 seconds ago, the queue is fully drained
        if last_io_time and idle_time > 15.0:
            print("\nAll enqueued points have been successfully processed and indexed (background queue empty)!")
            break
            
        if elapsed > 1800.0:  # Timeout after 30 minutes of indexing
            print("\nWarning: Indexing check timed out.")
            break
            
        time.sleep(5.0)

if __name__ == "__main__":
    main()
