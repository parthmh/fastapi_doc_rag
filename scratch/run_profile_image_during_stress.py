import subprocess
import time
import sys
import os

def main():
    # 1. Start run_locust_image_evaluation_csv.py in background
    print("Starting Locust image evaluation in the background...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "scratch/run_locust_image_evaluation_csv.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line buffered
        env=env
    )
    
    # 2. Monitor stdout for Locust start
    started_bombardment = False
    print("Waiting for Locust spawning to start...")
    
    # Read output line-by-line
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(f"[Locust Eval] {line.strip()}", flush=True)
        
        # Check if master started sending spawn jobs or completed spawning
        if "Sending spawn jobs of 6000 users" in line or "All users spawned" in line:
            started_bombardment = True
            print(">>> Heavy bombardment detected! Starting py-spy profiling now...", flush=True)
            break
            
    if not started_bombardment:
        print("Error: Locust did not start properly.", flush=True)
        proc.kill()
        return

    # 3. Wait 5 seconds into the bombardment to saturate the event loop
    time.sleep(5.0)

    # 4. Trigger py-spy command to capture the peak load (30 seconds duration)
    print("Running py-spy record for 30 seconds...", flush=True)
    py_spy_cmd = [
        "docker", "run", "--rm",
        "--pid=container:rag_backend",
        "--cap-add=SYS_PTRACE",
        "-v", f"{os.getcwd()}/processed:/outputs",
        "local-py-spy", "record",
        "-o", "/outputs/profile_gil_clean.svg",
        "--pid", "1",
        "--duration", "30",
        "--gil",
        "--subprocesses"
    ]
    
    spy_res = subprocess.run(py_spy_cmd, capture_output=True, text=True)
    if spy_res.returncode != 0:
        print(f"py-spy failed with error:\n{spy_res.stderr}", flush=True)
    else:
        print("py-spy completed successfully! SVG saved to processed/profile_gil_clean.svg", flush=True)

    # 5. Continue outputting from the background Locust process
    print("Continuing monitoring of Locust and backend queue drain...", flush=True)
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(f"[Locust Eval] {line.strip()}", flush=True)

    proc.wait()
    print("Evaluation run completed.", flush=True)

if __name__ == "__main__":
    main()
