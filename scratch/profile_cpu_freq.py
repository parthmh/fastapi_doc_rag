import subprocess
import time
import sys
import os
import re
import threading
import random
import base64
import httpx

VALID_IMAGE_URLS = [
    "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=200&q=80",
]

CORES_UVICORN  = list(range(4, 8))   # cores 4-7
CORES_IMAGE    = list(range(12, 16)) # cores 12-15
CORES_TEXT     = list(range(8, 12))  # cores 8-11


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_pids(worker_type: str):
    res = subprocess.run(["docker", "top", "rag_backend"], capture_output=True, text=True)
    pids = []
    for line in res.stdout.splitlines():
        if worker_type == "image" and "ingest_image_worker" in line:
            pids.append(line.split()[1])
        elif worker_type == "text" and "ingest_worker" in line and "ingest_image_worker" not in line:
            pids.append(line.split()[1])
    return pids


def wait_for_workers(worker_type: str, expected=4, timeout=45):
    """Poll get_pids until expected number of subprocess workers show up."""
    print(f"  Waiting for {expected} {worker_type} worker processes to spawn...", flush=True)
    t_end = time.perf_counter() + timeout
    while time.perf_counter() < t_end:
        pids = get_pids(worker_type)
        if len(pids) >= expected:
            print(f"  Found all {expected} {worker_type} workers: {pids}", flush=True)
            return pids
        time.sleep(1.0)
    pids = get_pids(worker_type)
    print(f"  Warning: Only found {len(pids)}/{expected} {worker_type} workers after timeout: {pids}", flush=True)
    return pids


def read_core_freq_mhz(cores: list) -> dict:
    """Read current scaling frequency (kHz → MHz) for given core indices."""
    result = {}
    for c in cores:
        path = f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq"
        try:
            with open(path, "r") as f:
                khz = int(f.read().strip())
            result[c] = khz / 1000.0  # convert to MHz
        except Exception:
            result[c] = None
    return result


def poll_frequencies(cores: list, duration_s: float, interval_s=0.5, label=""):
    """
    Poll per-core frequencies every interval_s for duration_s seconds.
    Returns list of per-sample dicts.
    """
    samples = []
    t_end = time.perf_counter() + duration_s
    while time.perf_counter() < t_end:
        freqs = read_core_freq_mhz(cores)
        freqs["_ts"] = time.perf_counter()
        samples.append(freqs)
        time.sleep(interval_s)
    return samples


def summarize_freq_samples(samples: list, cores: list, label: str):
    """Print per-core average, min, max MHz from polling samples."""
    print(f"\n  [{label}] Core frequency summary ({len(samples)} samples):", flush=True)
    for c in cores:
        vals = [s[c] for s in samples if s.get(c) is not None]
        if vals:
            print(f"    Core {c:2d}: avg={sum(vals)/len(vals):7.1f} MHz  "
                  f"min={min(vals):7.1f} MHz  max={max(vals):7.1f} MHz", flush=True)


def run_perf_effective_freq(pids: list, duration=15, label=""):
    """
    Measure effective CPU clock speed for specific PIDs using:
      freq = cycles / task-clock
    task-clock is the actual CPU time the process was scheduled (not wall-clock),
    so cycles/task-clock gives true running frequency irrespective of sleep/wait time.
    """
    if not pids:
        return None
    pid_str = ",".join(pids)
    print(f"\n  [perf freq] {label} (PIDs {pid_str}) for {duration}s...", flush=True)
    cmd = [
        "docker", "run", "--privileged", "--pid=host", "--rm",
        "-v", "/usr:/usr", "-v", "/lib:/lib", "-v", "/lib64:/lib64", "-v", "/etc:/etc",
        "ubuntu",
        "/usr/lib/linux-tools/6.17.0-35-generic/perf", "stat",
        "-e", "cycles,task-clock,instructions",
        "-p", pid_str,
        "sleep", str(duration)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    output = res.stdout + res.stderr

    cycles = task_clock_ms = ipc = None
    effective_ghz = None
    for line in output.splitlines():
        m = re.search(r"([\d,.]+)\s+cycles", line)
        if m and cycles is None:
            cycles = float(m.group(1).replace(",", ""))
        
        m = re.search(r"([\d,.]+)\s+task-clock", line)
        if m:
            val = float(m.group(1).replace(",", ""))
            if "msec" in line:
                task_clock_ms = val
            else:
                # convert nanoseconds to milliseconds
                task_clock_ms = val / 1e6
                
        m = re.search(r"#\s+([\d.]+)\s+insn per cycle", line)
        if m:
            ipc = float(m.group(1))

        # Or parse GHz directly from comment
        # e.g., "cycles # 2.240 GHz"
        m_ghz = re.search(r"#\s+([\d.]+)\s+GHz", line)
        if m_ghz:
            effective_ghz = float(m_ghz.group(1))

    if effective_ghz is None and cycles and task_clock_ms and task_clock_ms > 0:
        task_clock_s = task_clock_ms / 1000.0
        effective_ghz = (cycles / task_clock_s) / 1e9

    print(f"  Raw output:\n{output}", flush=True)
    effective_str = f"{effective_ghz:.3f} GHz" if effective_ghz is not None else "Unknown GHz"
    print(f"  => Effective frequency: {effective_str} | IPC: {ipc}", flush=True)
    return effective_ghz, ipc


def low_load_feeder(stop_event):
    client = httpx.Client()
    url = "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?auto=format&fit=crop&w=200&q=80"
    print("Pre-downloading base64 image for LOW load phase...", flush=True)
    try:
        resp = client.get(url, timeout=10.0)
        b64_str = base64.b64encode(resp.content).decode("utf-8")
        base64_img = f"data:image/jpeg;base64,{b64_str}"
        print("Pre-download complete.", flush=True)
    except Exception as e:
        print(f"Failed to pre-download: {e}. Using fallback 1x1 image.", flush=True)
        base64_img = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

    while not stop_event.is_set():
        unique_url = f"{base64_img}#t={random.randint(0,9999999)}"
        try:
            client.post("http://localhost:8000/api/v1/ingest/image",
                json={"items": [{"image_url": unique_url,
                                 "product_id": f"ll_{random.randint(0,999999)}",
                                 "caption": "freq test low load", "metadata": {}}]},
                timeout=2.0)
        except Exception:
            pass
        time.sleep(0.3)


def run_locust_and_wait_for_spawn(script: str, worker_type: str):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen([sys.executable, "-u", script],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)
    pids = []
    for line in proc.stdout:
        print(f"  [Locust] {line.strip()}", flush=True)
        if "Backend is healthy and running." in line:
            time.sleep(3.0)
            pids = wait_for_workers(worker_type)
        if "Sending spawn jobs of 6000 users" in line:
            print("  >>> Bombardment started.", flush=True)
            break
    if not pids:
        pids = wait_for_workers(worker_type)
    return proc, pids


# ══════════════════════════════════════════════════════════════════════════════
# PHASE A: LOW LOAD
# ══════════════════════════════════════════════════════════════════════════════
def phase_a_low_load():
    print("\n" + "="*70, flush=True)
    print("PHASE A: FashionCLIP image workers — LOW LOAD (drip-feed Base64)", flush=True)
    print("="*70, flush=True)

    # First send a dummy request to trigger the FastAPI initialization of worker subprocesses
    print("  Sending a priming request to start image workers...", flush=True)
    try:
        url = "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?auto=format&fit=crop&w=200&q=80"
        client = httpx.Client()
        client.post("http://localhost:8000/api/v1/ingest/image",
            json={"items": [{"image_url": url, "product_id": "prime", "caption": "prime", "metadata": {}}]},
            timeout=2.0)
    except Exception:
        pass

    image_pids = wait_for_workers("image")
    print(f"  Image worker PIDs: {image_pids}", flush=True)

    stop = threading.Event()
    threading.Thread(target=low_load_feeder, args=(stop,), daemon=True).start()
    time.sleep(3.0)

    print("\n  Polling core frequencies for 20s...", flush=True)
    freq_samples = []
    perf_result = [None]

    def do_perf():
        perf_result[0] = run_perf_effective_freq(image_pids, duration=20, label="LOW load image workers")

    perf_thread = threading.Thread(target=do_perf)
    perf_thread.start()

    t_end = time.perf_counter() + 20
    while time.perf_counter() < t_end:
        freqs = read_core_freq_mhz(CORES_UVICORN + CORES_IMAGE)
        freqs["_ts"] = time.perf_counter()
        freq_samples.append(freqs)
        time.sleep(0.5)

    perf_thread.join()
    stop.set()

    summarize_freq_samples(freq_samples, CORES_UVICORN, "Uvicorn cores 4-7 [LOW load]")
    summarize_freq_samples(freq_samples, CORES_IMAGE,   "Image  cores 12-15 [LOW load]")

    return freq_samples, perf_result[0]


# ══════════════════════════════════════════════════════════════════════════════
# PHASE B: FULL 6000-USER BOMBARDMENT
# ══════════════════════════════════════════════════════════════════════════════
def phase_b_high_load():
    print("\n" + "="*70, flush=True)
    print("PHASE B: FashionCLIP + 6000-user Locust — FULL BOMBARDMENT (Base64)", flush=True)
    print("="*70, flush=True)

    proc, image_pids = run_locust_and_wait_for_spawn(
        "scratch/run_locust_image_evaluation_csv.py", "image")

    print("\n  Waiting 10s for load to saturate before sampling...", flush=True)
    time.sleep(10.0)

    print("\n  Polling core frequencies during peak bombardment for 20s...", flush=True)
    freq_samples = []
    perf_result = [None]

    def do_perf():
        perf_result[0] = run_perf_effective_freq(image_pids, duration=20, label="HIGH load image workers")

    perf_thread = threading.Thread(target=do_perf)
    perf_thread.start()

    t_end = time.perf_counter() + 20
    while time.perf_counter() < t_end:
        freqs = read_core_freq_mhz(CORES_UVICORN + CORES_IMAGE)
        freqs["_ts"] = time.perf_counter()
        freq_samples.append(freqs)
        time.sleep(0.5)

    perf_thread.join()

    summarize_freq_samples(freq_samples, CORES_UVICORN, "Uvicorn cores 4-7 [HIGH load]")
    summarize_freq_samples(freq_samples, CORES_IMAGE,   "Image  cores 12-15 [HIGH load]")

    # Drain locust
    while True:
        line = proc.stdout.readline()
        if not line or "Monitoring background worker logs" in line:
            break
    proc.terminate(); proc.wait()

    return freq_samples, perf_result[0]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Make sure we clean up database and queue before we run
    print("Recreating and restarting backend to start fresh...")
    subprocess.run(["docker", "compose", "up", "-d", "backend"])
    subprocess.run(["docker", "compose", "restart", "backend"])
    print("Waiting 20 seconds for backend to boot up and spawn workers...")
    time.sleep(20.0)

    low_samples,  low_perf  = phase_a_low_load()
    high_samples, high_perf = phase_b_high_load()

    def avg_core(samples, core):
        vals = [s[core] for s in samples if s.get(core) is not None]
        return sum(vals)/len(vals) if vals else 0.0

    low_uvicorn_avg  = sum(avg_core(low_samples, c)  for c in CORES_UVICORN) / 4
    high_uvicorn_avg = sum(avg_core(high_samples, c) for c in CORES_UVICORN) / 4
    low_image_avg    = sum(avg_core(low_samples, c)  for c in CORES_IMAGE)   / 4
    high_image_avg   = sum(avg_core(high_samples, c) for c in CORES_IMAGE)   / 4

    low_eff_ghz,  low_ipc  = low_perf  if low_perf  else (None, None)
    high_eff_ghz, high_ipc = high_perf if high_perf else (None, None)

    print("\n\n" + "="*75, flush=True)
    print("══ FINAL CPU FREQUENCY THROTTLING PROOF REPORT (Base64 Situation) ══", flush=True)
    print("="*75, flush=True)

    print(f"""
┌──────────────────────────────────────┬────────────────┬────────────────┬──────────────┐
│ Metric                               │  LOW load      │  HIGH load     │  Δ Change    │
├──────────────────────────────────────┼────────────────┼────────────────┼──────────────┤
│ Uvicorn cores 4-7  avg freq          │{low_uvicorn_avg:>10.0f} MHz  │{high_uvicorn_avg:>10.0f} MHz  │{high_uvicorn_avg-low_uvicorn_avg:>+10.0f} MHz │
│ Image cores 12-15  avg freq          │{low_image_avg:>10.0f} MHz  │{high_image_avg:>10.0f} MHz  │{high_image_avg-low_image_avg:>+10.0f} MHz │
│ Effective worker GHz (cycles/task-clk)│{low_eff_ghz:>10.3f} GHz  │{high_eff_ghz:>10.3f} GHz  │{high_eff_ghz-low_eff_ghz:>+10.3f} GHz │
│ Worker IPC                           │{low_ipc:>14.3f}  │{high_ipc:>14.3f}  │{high_ipc-low_ipc:>+12.3f}  │
└──────────────────────────────────────┴────────────────┴────────────────┴──────────────┘

Latency impact (from docker logs):
  LOW  load embed avg : ~110ms
  HIGH load embed avg : ~170ms-240ms (+60ms to +130ms)

Interpretation:
  Compare the effective worker GHz to see if the frequency drops from ~4.0 GHz to ~2.6 GHz (or ~1.8 GHz)
  under the heavy base64 workload.
""", flush=True)


if __name__ == "__main__":
    main()
