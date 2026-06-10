import json
import random
from locust import HttpUser, task, between, events

class IngestUser(HttpUser):
    """
    Simulated user sending high-frequency requests, each containing exactly 1 chunk,
    to stress-test the concurrent ingestion endpoint.
    """
    # Simulate high-throughput requests with a minimal think time (10ms - 50ms)
    wait_time = between(0.01, 0.05)

    def on_start(self):
        try:
            with open("tests/synthetic_data.json", "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            print(f"[{self}] Loaded {len(self.chunks)} synthetic items for ingestion stress test.")
        except Exception as e:
            print(f"[{self}] Error loading synthetic_data.json: {e}")
            self.chunks = []

    @task
    def ingest_single_chunk(self):
        if not self.chunks:
            return

        # Select exactly 1 random chunk
        chunk = random.choice(self.chunks)
        payload = {
            "items": [chunk]
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        self.client.post("/api/v1/ingest", json=payload, headers=headers)
