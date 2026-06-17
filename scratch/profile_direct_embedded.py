import subprocess
import time
import sys
import os
import re

VALID_IMAGE_URLS = [
    "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1572635196237-14b3f281503f?auto=format&fit=crop&w=200&q=80"
]

def get_benchmark_pids():
    res = subprocess.run(["docker", "top", "rag_backend"], capture_output=True, text=True)
    pids = []
    for line in res.stdout.splitlines():
        if "benchmark_base64_direct.py" in line:
            parts = line.split()
            if len(parts) >= 2:
                pids.append(parts[1])
    return pids

def run_perf_dram(duration=10):
    print(f"  [DRAM Bandwidth] Measuring uncore IMC counters for {duration}s...", flush=True)
    cmd = [
        "docker", "run", "--privileged", "--pid=host", "--rm",
        "-v", "/usr:/usr", "-v", "/lib:/lib", "-v", "/lib64:/lib64",
        "-v", "/etc:/etc", "-v", "/sys:/sys",
        "ubuntu",
        "/usr/lib/linux-tools/6.17.0-35-generic/perf", "stat",
        "-e", ("uncore_imc_free_running_0/data_read/,"
               "uncore_imc_free_running_0/data_write/,"
               "uncore_imc_free_running_1/data_read/,"
               "uncore_imc_free_running_1/data_write/"),
        "-a", "sleep", str(duration)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    output = res.stdout + res.stderr
    
    ch0_r = ch0_w = ch1_r = ch1_w = 0.0
    for line in output.splitlines():
        m = re.search(r"([\d,]+\.?\d*)\s+MiB\s+uncore_imc_free_running_0/data_read/", line)
        if m: ch0_r = float(m.group(1).replace(",", ""))
        m = re.search(r"([\d,]+\.?\d*)\s+MiB\s+uncore_imc_free_running_0/data_write/", line)
        if m: ch0_w = float(m.group(1).replace(",", ""))
        m = re.search(r"([\d,]+\.?\d*)\s+MiB\s+uncore_imc_free_running_1/data_read/", line)
        if m: ch1_r = float(m.group(1).replace(",", ""))
        m = re.search(r"([\d,]+\.?\d*)\s+MiB\s+uncore_imc_free_running_1/data_write/", line)
        if m: ch1_w = float(m.group(1).replace(",", ""))
        
    total_read = ch0_r + ch1_r
    total_write = ch0_w + ch1_w
    read_bw = total_read / duration
    write_bw = total_write / duration
    total_bw_gb = (read_bw + write_bw) / 1024
    
    print(f"  => DRAM Read: {read_bw:.1f} MiB/s | Write: {write_bw:.1f} MiB/s | Total: {read_bw + write_bw:.1f} MiB/s ({total_bw_gb:.2f} GB/s)", flush=True)
    return read_bw, write_bw, total_bw_gb

def run_perf_pids(pids, duration=15):
    pid_str = ",".join(pids)
    print(f"  [PID Perf] Measuring L3, TLB, Cycles on PIDs [{pid_str}] for {duration}s...", flush=True)
    cmd = [
        "docker", "run", "--privileged", "--pid=host", "--rm",
        "-v", "/usr:/usr", "-v", "/lib:/lib", "-v", "/lib64:/lib64", "-v", "/etc:/etc",
        "ubuntu",
        "/usr/lib/linux-tools/6.17.0-35-generic/perf", "stat",
        "-e", "cycles,instructions,cache-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,dTLB-store-misses,iTLB-load-misses,task-clock,mem_inst_retired.stlb_miss_loads",
        "-p", pid_str,
        "sleep", str(duration)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    output = res.stdout + res.stderr
    
    metrics = {}
    for line in output.splitlines():
        line = line.strip()
        m = re.match(r"([\d,]+)\s+([\w.\-/]+)", line)
        if m:
            count = int(m.group(1).replace(",", ""))
            event = m.group(2)
            metrics[event] = count
            
    # Parse GHz and IPC from comments
    for line in output.splitlines():
        m = re.search(r"#\s+([\d.]+)\s+GHz", line)
        if m:
            metrics["_ghz"] = float(m.group(1))
        m = re.search(r"#\s+([\d.]+)\s+insn per cycle", line)
        if m:
            metrics["_ipc"] = float(m.group(1))
            
    return metrics

def run_experiment(num_workers):
    print(f"\n==========================================", flush=True)
    print(f"=== EXPERIMENT: {num_workers} WORKER(S) DIRECT ===", flush=True)
    print(f"==========================================", flush=True)
    
    procs = []
    iterations = 220 if num_workers == 4 else 120
    for i in range(num_workers):
        core = 12 + i
        cmd = [
            "docker", "exec", "rag_backend",
            "taskset", "-c", str(core),
            "python", "/app/processed/benchmark_base64_direct.py",
            "--iterations", str(iterations),
            "--worker-id", str(i)
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append(p)
        print(f"Spawned worker {i} on Core {core}.", flush=True)
        
    time.sleep(8.0) # wait for startup
    
    pids = get_benchmark_pids()
    print(f"Active benchmark PIDs: {pids}", flush=True)
    if not pids:
        print("Error: Could not get PIDs", flush=True)
        return None
        
    read_bw, write_bw, total_bw = run_perf_dram(duration=10)
    metrics = run_perf_pids(pids, duration=12)
    
    avg_latencies = []
    for i, p in enumerate(procs):
        stdout, stderr = p.communicate()
        avg = None
        for line in stdout.splitlines():
            if "Average embedding latency:" in line:
                avg = float(line.split(":")[-1].strip().split()[0])
        if avg:
            avg_latencies.append(avg)
            
    avg_lat = sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0.0
    print(f"Experiment finished. Average embedding latency: {avg_lat:.2f} ms", flush=True)
    
    metrics["_read_bw"] = read_bw
    metrics["_write_bw"] = write_bw
    metrics["_total_bw"] = total_bw
    metrics["_latency"] = avg_lat
    
    # Calculate rates
    loads = metrics.get("LLC-loads", 0)
    misses = metrics.get("LLC-load-misses", 0)
    metrics["_l3_rate"] = (misses / loads * 100) if loads > 0 else 0.0
    
    dtlb_loads = metrics.get("dTLB-loads", 0)
    dtlb_misses = metrics.get("dTLB-load-misses", 0)
    metrics["_dtlb_rate"] = (dtlb_misses / dtlb_loads * 100) if dtlb_loads > 0 else 0.0
    
    return metrics

def main():
    exp1 = run_experiment(1)
    time.sleep(5.0)
    exp4 = run_experiment(4)
    
    print("\n\n" + "="*80, flush=True)
    print("══ REPLICATED MICROARCHITECTURAL COMPARISON REPORT ══", flush=True)
    print("="*80, flush=True)
    
    def get_val(data, key, format_str="{:,}", default=0):
        if not data or key not in data: return "N/A"
        return format_str.format(data[key])
        
    def get_float(data, key, format_str="{:.3f}", default=0.0):
        if not data or key not in data: return "N/A"
        return format_str.format(data[key])

    print(f"""
┌─────────────────────────────────────────────┬───────────────────┬───────────────────┐
│ Metric                                      │    1 Worker       │    4 Workers      │
├─────────────────────────────────────────────┼───────────────────┼───────────────────┤
│ Effective Freq (GHz)                        │ {get_float(exp1, "_ghz"):>17}   │ {get_float(exp4, "_ghz"):>17}   │
│ IPC (Instructions Per Cycle)                │ {get_float(exp1, "_ipc"):>17}   │ {get_float(exp4, "_ipc"):>17}   │
├─────────────────────────────────────────────┼───────────────────┼───────────────────┤
│ LLC Loads (L3 accesses)                     │ {get_val(exp1, "LLC-loads"):>17}   │ {get_val(exp4, "LLC-loads"):>17}   │
│ LLC Load Misses (L3 misses)                 │ {get_val(exp1, "LLC-load-misses"):>17}   │ {get_val(exp4, "LLC-load-misses"):>17}   │
│ LLC Load Miss Rate (%)                      │ {get_float(exp1, "_l3_rate", "{:.2f}%"):>17}   │ {get_float(exp4, "_l3_rate", "{:.2f}%"):>17}   │
├─────────────────────────────────────────────┼───────────────────┼───────────────────┤
│ DRAM Read Bandwidth (MiB/s)                │ {get_float(exp1, "_read_bw", "{:.1f}"):>17}   │ {get_float(exp4, "_read_bw", "{:.1f}"):>17}   │
│ DRAM Write Bandwidth (MiB/s)               │ {get_float(exp1, "_write_bw", "{:.1f}"):>17}   │ {get_float(exp4, "_write_bw", "{:.1f}"):>17}   │
│ DRAM Total Bandwidth (GB/s)                 │ {get_float(exp1, "_total_bw", "{:.2f}"):>17}   │ {get_float(exp4, "_total_bw", "{:.2f}"):>17}   │
├─────────────────────────────────────────────┼───────────────────┼───────────────────┤
│ dTLB Loads                                  │ {get_val(exp1, "dTLB-loads"):>17}   │ {get_val(exp4, "dTLB-loads"):>17}   │
│ dTLB Load Misses                            │ {get_val(exp1, "dTLB-load-misses"):>17}   │ {get_val(exp4, "dTLB-load-misses"):>17}   │
│ dTLB Load Miss Rate (%)                     │ {get_float(exp1, "_dtlb_rate", "{:.3f}%"):>17}   │ {get_float(exp4, "_dtlb_rate", "{:.3f}%"):>17}   │
│ dTLB Store Misses                           │ {get_val(exp1, "dTLB-store-misses"):>17}   │ {get_val(exp4, "dTLB-store-misses"):>17}   │
│ iTLB Load Misses                            │ {get_val(exp1, "iTLB-load-misses"):>17}   │ {get_val(exp4, "iTLB-load-misses"):>17}   │
│ STLB Miss Loads                             │ {get_val(exp1, "mem_inst_retired.stlb_miss_loads"):>17}   │ {get_val(exp4, "mem_inst_retired.stlb_miss_loads"):>17}   │
├─────────────────────────────────────────────┼───────────────────┼───────────────────┤
│ Average Embedding Latency                   │ {get_float(exp1, "_latency", "{:.2f} ms"):>17}   │ {get_float(exp4, "_latency", "{:.2f} ms"):>17}   │
└─────────────────────────────────────────────┴───────────────────┴───────────────────┘
""", flush=True)

if __name__ == "__main__":
    main()
