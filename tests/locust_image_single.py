import random
import base64
import httpx
from locust import HttpUser, task, constant, events

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

FALLBACK_B64 = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
BASE64_IMAGE_STRINGS = []

# Pre-download images on module import
print("Pre-downloading images and converting to base64...", flush=True)
for url in VALID_IMAGE_URLS:
    try:
        resp = httpx.get(url, timeout=15.0)
        resp.raise_for_status()
        b64_str = base64.b64encode(resp.content).decode("utf-8")
        BASE64_IMAGE_STRINGS.append(f"data:image/jpeg;base64,{b64_str}")
    except Exception as e:
        print(f"Failed to pre-download {url}: {e}", flush=True)

if not BASE64_IMAGE_STRINGS:
    print("WARNING: All pre-downloads failed. Using fallback 1x1 image.", flush=True)
    BASE64_IMAGE_STRINGS.append(FALLBACK_B64)
else:
    print(f"Successfully pre-downloaded and converted {len(BASE64_IMAGE_STRINGS)} images.", flush=True)


class IngestImageSingleUser(HttpUser):
    """
    Simulated user sending high-frequency image ingestion requests,
    each containing exactly 1 base64 encoded image, to stress-test the pipeline.
    """
    # No think time to bombard the server as fast as possible
    wait_time = constant(0)

    @task
    def ingest_single_image(self):
        product_id = f"prod_{random.randint(1000000, 9999999)}"
        base_b64 = random.choice(BASE64_IMAGE_STRINGS)
        # Append unique hash fragment so each request produces a unique point ID in Qdrant
        unique_url = f"{base_b64}#locust_idx={random.randint(0, 10000000)}"
        
        payload = {
            "items": [
                {
                    "image_url": unique_url,
                    "product_id": product_id,
                    "caption": f"Mock fashion item {product_id}",
                    "metadata": {
                        "color": random.choice(["red", "blue", "black", "white", "green"]),
                        "source": "locust_stress_base64"
                    }
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        self.client.post("/api/v1/ingest/image", json=payload, headers=headers)


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"--- Starting Locust Image Ingestion Stress Test against target: {environment.host} ---", flush=True)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("--- Completed Locust Image Ingestion Stress Test ---", flush=True)
