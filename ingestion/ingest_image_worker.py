from __future__ import annotations

import sys
import os
import select
import time
import concurrent.futures
import uuid
from io import BytesIO
from typing import Any

import httpx
import orjson
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient, models

from app.config import settings
from app.schemas import IngestImageItem
from ingestion.embed_core import stable_point_id

# Singletons for lazy loading
_fashion_clip_model: CLIPModel | None = None
_fashion_clip_processor: CLIPProcessor | None = None

def get_fashion_clip_model() -> tuple[CLIPModel, CLIPProcessor]:
    global _fashion_clip_model, _fashion_clip_processor
    if _fashion_clip_model is None:
        print("Loading FashionCLIP model ('patrickjohncyh/fashion-clip') on CPU...", flush=True)
        # Enforce PyTorch to use 1 thread to avoid core scheduling conflicts
        torch.set_num_threads(1)
        _fashion_clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        _fashion_clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    return _fashion_clip_model, _fashion_clip_processor

def ensure_image_collection_initialized(client: QdrantClient) -> None:
    collection_name = settings.image_collection_name
    if client.collection_exists(collection_name):
        return

    print(f"Creating image collection '{collection_name}'...", flush=True)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=512,  # FashionCLIP returns 512-dimensional embeddings
            distance=models.Distance.COSINE,
        )
    )
    # create payload index on product_id
    client.create_payload_index(
        collection_name=collection_name,
        field_name="product_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

def download_image(url: str) -> Image.Image:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    resp = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")

def download_images_concurrently(urls: list[str]) -> list[Image.Image | None]:
    results = [None] * len(urls)
    # Use ThreadPoolExecutor to handle concurrent network I/O
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_index = {
            executor.submit(download_image, url): i
            for i, url in enumerate(urls)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"Error downloading image from {urls[i]}: {e}", flush=True)
                results[i] = None
    return results

def process_image_batch(batch: list[IngestImageItem], client: QdrantClient) -> None:
    if not batch:
        return

    urls = [item.image_url for item in batch]
    
    # 1. Concurrent downloads (I/O-Bound)
    start_download = time.perf_counter()
    images = download_images_concurrently(urls)
    download_latency = time.perf_counter() - start_download

    # Filter out failed downloads
    valid_items = []
    valid_images = []
    for item, img in zip(batch, images):
        if img is not None:
            valid_items.append(item)
            valid_images.append(img)
            
    if not valid_images:
        print(f"All downloads in batch of {len(batch)} failed. Skipping database upsert.", flush=True)
        return

    # 2. Embedding generation (CPU-Bound)
    start_embed = time.perf_counter()
    model, processor = get_fashion_clip_model()
    
    # Process images to tensors
    inputs = processor(images=valid_images, return_tensors="pt")
    
    # Run CPU inference
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        # Handle cases where return_dict is True and returns a BaseModelOutputWithPooling object
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output
        # L2 normalize the features to match COSINE distance metric
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embeddings = image_features.cpu().numpy().tolist()
        
    embed_latency = time.perf_counter() - start_embed

    # 3. Point Construction & Bulk Upsert
    start_upsert = time.perf_counter()
    collection_name = settings.image_collection_name
    points = []
    
    for item, vector in zip(valid_items, embeddings, strict=True):
        point_id = stable_point_id(str(item.image_url))
        payload = {
            "product_id": item.product_id,
            "image_url": str(item.image_url),
            "caption": item.caption,
            "metadata": item.metadata or {},
        }
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    # Execute non-blocking upsert
    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=False,
    )
    upsert_latency = time.perf_counter() - start_upsert
    
    print(
        f"Batch processed: {len(valid_images)} succeeded, {len(batch) - len(valid_images)} failed | "
        f"Download: {download_latency:.4f}s | Embed: {embed_latency:.4f}s | Upsert: {upsert_latency:.4f}s",
        flush=True
    )

def run_subprocess_worker() -> None:
    # Initialize Qdrant Client in child process
    client = QdrantClient(url=settings.qdrant_url)
    
    # Auto-create collection if not exists
    ensure_image_collection_initialized(client)
    
    print("Subprocess Image Ingestion Worker initialized.", flush=True)
    
    batch_size = settings.ingest_batch_size
    batch_raw = []
    buffer = b""
    
    while True:
        try:
            # 1. If batch is empty, block until we read something
            if not batch_raw:
                chunk = os.read(0, 65536)
                if not chunk:
                    # EOF reached
                    break
                buffer += chunk
            else:
                # If batch is not empty, check if we can read more without blocking
                r, _, _ = select.select([0], [], [], 0)
                if r:
                    chunk = os.read(0, 65536)
                    if not chunk:
                        # EOF
                        if batch_raw:
                            batch = [IngestImageItem(**item) for item in batch_raw]
                            process_image_batch(batch, client)
                        break
                    buffer += chunk
                else:
                    # No more data immediately available, process current batch
                    batch = [IngestImageItem(**item) for item in batch_raw]
                    process_image_batch(batch, client)
                    batch_raw = []
                    continue

            # Process complete lines from buffer
            has_sentinel = False
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line:
                    continue
                item_raw = orjson.loads(line)
                if item_raw is None:
                    has_sentinel = True
                    break
                batch_raw.append(item_raw)
                if len(batch_raw) >= batch_size:
                    batch = [IngestImageItem(**item) for item in batch_raw]
                    process_image_batch(batch, client)
                    batch_raw = []

            if has_sentinel:
                if batch_raw:
                    batch = [IngestImageItem(**item) for item in batch_raw]
                    process_image_batch(batch, client)
                print("Received shutdown sentinel. Exiting image worker.", flush=True)
                break
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error in subprocess image worker loop: {e}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    run_subprocess_worker()
