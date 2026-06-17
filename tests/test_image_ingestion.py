import time
import httpx

API_URL = "http://localhost:8000/api/v1/ingest/image"
QDRANT_COLLECTION_URL = "http://localhost:6333/collections/fashion_images_fashion_clip"

def run_test():
    print("Sending mock image ingestion request to FastAPI API...")
    payload = {
        "items": [
            {
                "image_url": "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?auto=format&fit=crop&w=200&q=80",
                "product_id": "prod_tshirt_001",
                "caption": "A classic linen blue shirt",
                "metadata": {"color": "blue", "category": "shirt"}
            },
            {
                "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=200&q=80",
                "product_id": "prod_shoe_002",
                "caption": "A pair of red running shoes",
                "metadata": {"color": "red", "category": "shoes"}
            }
        ]
    }

    resp = httpx.post(API_URL, json=payload, timeout=10.0)
    print(f"API Response status: {resp.status_code}")
    print(f"API Response body: {resp.json()}")

    assert resp.status_code == 202, f"Expected 202, got {resp.status_code}"
    
    # Wait for the background worker to download the images, load the FashionCLIP model, and upsert them to Qdrant
    print("Waiting 30 seconds for background worker to download, embed, and upsert...")
    time.sleep(30)

    print("Checking Qdrant DB for points in 'fashion_images_fashion_clip' collection...")
    resp_qdrant = httpx.post(
        f"{QDRANT_COLLECTION_URL}/points/scroll",
        json={"limit": 10, "with_vector": True, "with_payload": True},
        timeout=10.0
    )
    
    print(f"Qdrant response status: {resp_qdrant.status_code}")
    assert resp_qdrant.status_code == 200, f"Expected 200 from Qdrant, got {resp_qdrant.status_code}"
    
    result = resp_qdrant.json().get("result", {})
    points = result.get("points", [])
    print(f"Number of points found in Qdrant: {len(points)}")
    
    assert len(points) > 0, "No points were indexed in Qdrant!"
    
    for pt in points:
        payload = pt.get("payload", {})
        vector = pt.get("vector", [])
        print(f"Point ID: {pt.get('id')}")
        print(f"Payload: {payload}")
        print(f"Vector dimension: {len(vector)}")
        assert len(vector) == 512, f"Expected 512-dimensional embedding, got {len(vector)}"

    print("TEST PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_test()
