import json
import random
from locust import HttpUser, task, constant, events

# Load synthetic dataset once at the module level so all simulated users share the same list.
# This avoids copying the 20,000-item dataset per user, preventing out-of-memory (OOM) crashes.
try:
    with open("tests/synthetic_data.json", "r", encoding="utf-8") as f:
        SHARED_CHUNKS = json.load(f)
    print(f"Loaded {len(SHARED_CHUNKS)} shared synthetic items for load test.")
except Exception as e:
    print(f"Error loading synthetic_data.json: {e}")
    SHARED_CHUNKS = []

class IngestUser(HttpUser):
    """
    Simulated user sending high-frequency requests, each containing exactly 1 chunk,
    to stress-test the concurrent ingestion endpoint.
    """
    # No think time to bombard the server as fast as possible
    wait_time = constant(0)

    @task
    def ingest_single_chunk(self):
        if not SHARED_CHUNKS:
            return

        # Select exactly 1 random chunk from shared memory
        chunk = random.choice(SHARED_CHUNKS)
        payload = {
            "items": [chunk]
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        self.client.post("/api/v1/ingest", json=payload, headers=headers)
