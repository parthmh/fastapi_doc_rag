import asyncio
import time
import httpx
import argparse
import sys

API_URL = "http://localhost:8000/api/v1/ingest"
QDRANT_URL = "http://localhost:6333/collections/fastapi_doc_ingest_minilm"

def generate_mock_batch(batch_index: int, batch_size: int) -> dict:
    """
    Generates a batch of mock chunks on the fly.
    """
    items = []
    for i in range(batch_size):
        idx = batch_index * batch_size + i
        items.append({
            "chunk_text": f"{idx}: This is mock paragraph number {idx} for benchmarking high concurrency ingestion.",
            "heading_text": f"Benchmark Section {idx // 10}",
            "page_id": f"benchmark/page_{idx // 100}",
            "section_url": f"https://example.com/benchmark/page_{idx // 100}#section-{idx}"
        })
    return {"items": items}

async def send_batch(client: httpx.AsyncClient, batch_index: int, batch_size: int, sem: asyncio.Semaphore) -> float:
    """
    Sends a batch of chunks to the API.
    """
    payload = generate_mock_batch(batch_index, batch_size)
    async with sem:
        start_time = time.perf_counter()
        try:
            resp = await client.post(API_URL, json=payload, timeout=60.0)
            latency = time.perf_counter() - start_time
            if resp.status_code == 202:
                return latency
            else:
                print(f"Error sending batch {batch_index}: Status {resp.status_code} - {resp.text}")
                return -1.0
        except Exception as e:
            print(f"Exception sending batch {batch_index}: {e}")
            return -1.0

async def get_qdrant_points_count(client: httpx.AsyncClient) -> int:
    """
    Query Qdrant to get the current points count.
    """
    try:
        resp = await client.get(QDRANT_URL)
        if resp.status_code == 200:
            res_json = resp.json()
            return res_json.get("result", {}).get("points_count", 0)
    except Exception as e:
        print(f"Error querying Qdrant: {e}")
    return 0

async def run_benchmark(target_points: int, batch_size: int, concurrency: int):
    print("=" * 70)
    print(f"Starting Ingestion Pipeline Benchmark: {target_points:,} Points")
    print(f"API Target: {API_URL}")
    print(f"Batch Size: {batch_size} | Concurrent Connections: {concurrency}")
    print("=" * 70)

    # Initialize Qdrant count
    async with httpx.AsyncClient() as client:
        start_points = await get_qdrant_points_count(client)
        print(f"Initial Qdrant points count: {start_points:,}")

    num_batches = target_points // batch_size
    sem = asyncio.Semaphore(concurrency)

    start_time = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(num_batches):
            tasks.append(send_batch(client, i, batch_size, sem))
        
        print(f"Enqueuing {num_batches} batches to the API...")
        results = await asyncio.gather(*tasks)
        
        # Filter successful latencies
        latencies = [l for l in results if l > 0]
        
    total_time = time.perf_counter() - start_time
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    throughput_api = target_points / total_time

    print("\n" + "=" * 70)
    print("--- API Enqueuing Phase Completed ---")
    print(f"Total Time to Accept all Points: {total_time:.4f} seconds")
    print(f"API Throughput: {throughput_api:.2f} points/second")
    print(f"Average Route Response Latency: {avg_latency * 1000:.2f} ms")
    print("=" * 70)

    # Now monitor background processing/upserting to Qdrant
    print("\nMonitoring background vector generation and Qdrant indexing...")
    print("Press Ctrl+C to stop monitoring once target is reached.")
    print(f"{'Elapsed (s)':<12}{'Qdrant Points':<18}{'Ingested (Net)':<18}{'Upsert Rate (pts/s)':<20}")
    print("-" * 70)

    bg_start_time = time.perf_counter()
    async with httpx.AsyncClient() as client:
        while True:
            current_points = await get_qdrant_points_count(client)
            net_ingested = current_points - start_points
            elapsed = time.perf_counter() - bg_start_time
            rate = net_ingested / elapsed if elapsed > 0 else 0
            
            print(f"{elapsed:<12.1f}{current_points:<18,}{net_ingested:<18,}{rate:<20.2f}")
            
            if net_ingested >= target_points:
                print("\nTarget point ingestion limit reached in Qdrant!")
                break
                
            await asyncio.sleep(5.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ingestion of up to 1M points.")
    parser.add_argument("--count", "-c", type=int, default=100000, help="Number of points to ingest")
    parser.add_argument("--batch", "-b", type=int, default=1, help="Batch size per request")
    parser.add_argument("--concurrency", type=int, default=15, help="Number of concurrent connections")
    args = parser.parse_args()
    
    try:
        asyncio.run(run_benchmark(args.count, args.batch, args.concurrency))
    except KeyboardInterrupt:
        print("\nBenchmark monitoring terminated by user.")
