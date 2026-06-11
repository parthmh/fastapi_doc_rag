import time
import torch
from sentence_transformers import SentenceTransformer

def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name, device="cpu")
    
    # Generate 64 items (1 batch)
    dummy_texts = [
        f"This is document number {i} containing some sample text to benchmark embedding generation speed."
        for i in range(64)
    ]
    
    print("=== PyTorch CPU Thread Count Benchmarking (Batch Size 64) ===")
    for threads in [1, 2, 4, 6, 8]:
        torch.set_num_threads(threads)
        
        # Warmup
        model.encode(dummy_texts, batch_size=64)
        
        # Benchmark 5 runs
        start = time.perf_counter()
        for _ in range(5):
            model.encode(dummy_texts, batch_size=64)
        duration = (time.perf_counter() - start) / 5.0 * 1000.0
        
        print(f"Threads: {threads} | Avg Latency per batch of 64: {duration:.2f} ms | Throughput: {64/(duration/1000.0):.2f} pts/sec")

if __name__ == "__main__":
    main()
