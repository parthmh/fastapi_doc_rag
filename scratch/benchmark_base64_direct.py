import base64
import time
import httpx
import torch
from io import BytesIO
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

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

def main():
    # Enforce PyTorch single thread to match ingestion environment
    torch.set_num_threads(1)
    
    # 1. Pre-download images and convert to base64
    print("Pre-downloading images and converting to base64...", flush=True)
    base64_strings = []
    for url in VALID_IMAGE_URLS:
        try:
            resp = httpx.get(url, timeout=15.0)
            resp.raise_for_status()
            b64_str = base64.b64encode(resp.content).decode("utf-8")
            base64_strings.append(f"data:image/jpeg;base64,{b64_str}")
        except Exception as e:
            print(f"Failed to download {url}: {e}", flush=True)
            
    if not base64_strings:
        print("Error: No images could be downloaded.", flush=True)
        return
        
    print(f"Successfully cached {len(base64_strings)} base64 images in memory.\n", flush=True)
    
    # 2. Load model and processor
    print("Loading CLIPModel and CLIPProcessor...", flush=True)
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    print("Model and processor loaded.\n", flush=True)
    
    # Warmup pass
    print("Warming up model with 1 pass...", flush=True)
    dummy_tensor = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        _ = model.get_image_features(pixel_values=dummy_tensor)
    print("Warmup complete.\n", flush=True)
    
    # 3. Timed execution loop
    print("Running 20 timed direct Base64 inferences...", flush=True)
    latencies = []
    for i in range(20):
        b64_str = base64_strings[i % len(base64_strings)]
        
        start_time = time.perf_counter()
        
        # Decode base64 to PIL Image
        header, base64_data = b64_str.split(",", 1)
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            _ = model.get_image_features(pixel_values=inputs["pixel_values"])
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(elapsed_ms)
        print(f"Pass {i+1:02d}: {elapsed_ms:.2f} ms", flush=True)
        
    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage embedding latency: {avg_latency:.2f} ms", flush=True)

if __name__ == "__main__":
    main()
