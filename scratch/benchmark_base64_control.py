import subprocess
import time
import sys

def main():
    print("Starting concurrent standalone benchmark with 4 workers on Cores 12-15...", flush=True)
    processes = []
    
    # Spawn 4 workers on Cores 12-15
    for i in range(4):
        core = 12 + i
        cmd = [
            "docker", "exec", "rag_backend",
            "taskset", "-c", str(core),
            "python", "/app/processed/benchmark_base64_direct.py",
            "--iterations", "200",
            "--worker-id", str(i)
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        processes.append(proc)
        print(f"Spawned Worker {i} on Core {core}.", flush=True)
        
    # Monitor outputs concurrently in a non-blocking loop
    import select
    
    # We will poll stdout of all processes
    streams = {p.stdout: i for i, p in enumerate(processes)}
    
    while streams:
        # Wait for data on any of the stdout streams
        rlist, _, _ = select.select(list(streams.keys()), [], [])
        for stream in rlist:
            line = stream.readline()
            if line:
                worker_id = streams[stream]
                print(f"[W{worker_id}] {line.strip()}", flush=True)
            else:
                # End of stream, remove it
                del streams[stream]
                
    # Wait for all processes to complete
    for i, proc in enumerate(processes):
        proc.wait()
        
    print("\nBenchmark finished.", flush=True)

if __name__ == "__main__":
    main()
