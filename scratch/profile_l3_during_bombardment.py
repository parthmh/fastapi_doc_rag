import subprocess
import time
import sys
import os
import re

def get_image_worker_pids():
    res = subprocess.run(["docker", "top", "rag_backend"], capture_output=True, text=True)
    pids = []
    for line in res.stdout.splitlines():
        if "ingest_image_worker" in line:
            parts = line.split()
            if len(parts) >= 2:
                pids.append(parts[1])
    return pids

def run_perf_stat(pids, duration=15):
    pid_str = ",".join(pids)
    print(f"Attaching perf stat to image worker PIDs [{pid_str}] for {duration}s...", flush=True)
    perf_cmd = [
        "docker", "run", "--privileged", "--pid=host", "--rm",
        "-v", "/usr:/usr",
        "-v", "/lib:/lib",
        "-v", "/lib64:/lib64",
        "-v", "/etc:/etc",
        "ubuntu",
        "/usr/lib/linux-tools/6.17.0-35-generic/perf", "stat",
        "-e", "cache-misses,LLC-loads,LLC-load-misses,instructions,cycles,stalled-cycles-backend",
        "-p", pid_str,
        "sleep", str(duration)
    ]
    perf_res = subprocess.run(perf_cmd, capture_output=True, text=True)
    return perf_res.stderr

def main():
    # 1. Start the main Locust evaluation script in the background
    print("Starting Locust evaluation in the background...", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "scratch/run_locust_image_evaluation_csv.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )
    
    # 2. Wait for bombardment to start and get PIDs
    pids = []
    started_bombardment = False
    print("Monitoring output for bombardment start...", flush=True)
    
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(f"[Locust Eval] {line.strip()}", flush=True)
        
        # Once backend is healthy and running, get the PIDs (since container was restarted)
        if "Backend is healthy and running." in line:
            time.sleep(2.0)
            pids = get_image_worker_pids()
            print(f"Detected active container image worker host PIDs: {pids}", flush=True)
            
        if "Sending spawn jobs of 6000 users" in line:
            started_bombardment = True
            print("\n>>> Heavy bombardment started! Waiting 15 seconds to saturate load...", flush=True)
            break
            
    if not started_bombardment or not pids:
        print("Error: Bombardment or PIDs could not be detected.", flush=True)
        proc.kill()
        return

    # 3. Wait 15s to saturate the system
    time.sleep(15.0)
    
    # 4. Profile during active bombardment
    print("\n>>> PROFILING ACTIVE BOMBARDMENT PHASE <<<", flush=True)
    bombardment_stats = run_perf_stat(pids, duration=15)
    
    # 5. Let the evaluation finish spawning and print results
    print("\nWaiting for Locust run to complete and transition to queue drain...", flush=True)
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(f"[Locust Eval] {line.strip()}", flush=True)
        
        if "Monitoring background worker logs to detect queue completion..." in line:
            print("\n>>> Cooldown/Drain phase detected! Waiting 10 seconds...", flush=True)
            break
            
    # Wait 10s to settle
    time.sleep(10.0)
    
    # 6. Profile during quiet queue drain phase
    print("\n>>> PROFILING QUIET QUEUE DRAIN PHASE <<<", flush=True)
    drain_stats = run_perf_stat(pids, duration=15)
    
    # 7. Print remaining Locust log lines
    print("\nDraining remaining Locust log lines...", flush=True)
    proc.terminate()
    proc.wait()
    
    print("\n=============================================", flush=True)
    print("=== FINAL HARDWARE PROFILE COMPARISON ===", flush=True)
    print("=============================================", flush=True)
    print("\n>>> ACTIVE BOMBARDMENT STATS (Contention) <<<", flush=True)
    print(bombardment_stats, flush=True)
    print("\n>>> QUIET QUEUE DRAIN STATS (No Contention) <<<", flush=True)
    print(drain_stats, flush=True)

if __name__ == "__main__":
    main()
