import json
import random
from locust import HttpUser, task, between, events

class IngestUser(HttpUser):
    """
    Simulated ingestion service client posting batches of synthetic documentation chunks.
    """
    # Wait time between request bursts to simulate standard ingestion workloads
    wait_time = between(0.1, 0.5)

    def on_start(self):
        """
        Load the synthetic dataset once at client startup.
        """
        try:
            with open("tests/synthetic_data.json", "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            print(f"[{self}] Loaded {len(self.chunks)} synthetic items for ingestion.")
        except Exception as e:
            print(f"[{self}] Error loading synthetic_data.json: {e}")
            self.chunks = []

    @task
    def ingest_batches(self):
        """
        Post a random batch of 10-50 synthetic documentation nodes.
        """
        if not self.chunks:
            return

        batch_size = random.randint(10, 50)
        batch = random.sample(self.chunks, min(batch_size, len(self.chunks)))

        payload = {
            "items": batch
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        with self.client.post("/api/v1/ingest", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 202:
                try:
                    res_json = response.json()
                    if res_json.get("status") == "accepted" and "task_id" in res_json:
                        response.success()
                    else:
                        response.failure("Response schema mismatch or missing task_id.")
                except Exception as e:
                    response.failure(f"Failed to parse response JSON: {e}")
            else:
                response.failure(f"Request failed with status {response.status_code}: {response.text}")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"--- Starting Locust Ingestion Load Test against target: {environment.host} ---")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("--- Completed Locust Ingestion Load Test ---")
