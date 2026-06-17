import random
from locust import HttpUser, task, between, events

VALID_IMAGE_URLS = [
    "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1572635196237-14b3f281503f?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1491553895911-0055eca6402d?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1560343090-f0409e92791a?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1524758631624-e2822e304c36?auto=format&fit=crop&w=200&q=80",
    "https://images.unsplash.com/photo-1511556532299-8f662fc26c06?auto=format&fit=crop&w=200&q=80"
]

class IngestImageUser(HttpUser):
    """
    Simulated ingestion client posting batches of fashion image metadata to the API.
    """
    # Wait time between request bursts to simulate standard ingestion workloads
    wait_time = between(0.1, 0.5)

    @task
    def ingest_image_batches(self):
        """
        Post a random batch of 5-20 image items.
        """
        batch_size = random.randint(5, 20)
        items = []
        for i in range(batch_size):
            # Pick a random image URL and generate a random product ID
            url = random.choice(VALID_IMAGE_URLS)
            product_id = f"prod_{random.randint(1000, 9999)}"
            # Avoid URL collisions in same batch by adding index query parameter
            unique_url = f"{url}&locust_idx={random.randint(0, 1000000)}"
            items.append({
                "image_url": unique_url,
                "product_id": product_id,
                "caption": f"Mock fashion item {product_id}",
                "metadata": {"color": random.choice(["red", "blue", "black", "white"]), "source": "locust"}
            })

        payload = {
            "items": items
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        with self.client.post("/api/v1/ingest/image", json=payload, headers=headers, catch_response=True) as response:
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
    print(f"--- Starting Locust Image Ingestion Load Test against target: {environment.host} ---")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("--- Completed Locust Image Ingestion Load Test ---")
