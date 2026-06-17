
import sys
import orjson
import select
import time

print("Child started", flush=True)

batch_size = 5
batch = []

while True:
    # 1. Block waiting for the first item
    line = sys.stdin.buffer.readline()
    if not line:
        print("Child: EOF on stdin. Exiting.", flush=True)
        break
    
    item = orjson.loads(line)
    if item is None:
        print("Child: Received shutdown sentinel. Exiting.", flush=True)
        break
        
    batch.append(item)
    
    # 2. Drain any other immediately available items without blocking
    while len(batch) < batch_size:
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            line2 = sys.stdin.buffer.readline()
            if not line2:
                break
            item2 = orjson.loads(line2)
            if item2 is None:
                # We got a shutdown sentinel, we should handle it
                # For this test, we can just process current batch and exit
                break
            batch.append(item2)
        else:
            break
            
    print(f"Child processed batch of size {len(batch)}: {batch}", flush=True)
    batch = []
