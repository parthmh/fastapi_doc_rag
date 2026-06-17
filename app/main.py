from typing import Annotated
import time
import os
import asyncio
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Body, Response, Depends, Request
from fastapi.staticfiles import StaticFiles
from anyio.to_thread import run_sync

from app.config import settings
from app.retriever import Retriever
from app.context_builder import reconstruct_context
from app.llm_client import generate_llm_response
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    RetrievedDocMetadata,
    ChatResponseMetadata,
    TokenUsage,
    ErrorResponse,
    IngestRequest,
    IngestResponse,
    IngestImageRequest,
    IngestImageResponse,
)
from ingestion.ingest_worker import ensure_ingest_collection_initialized, ingest_worker_loop

tags_metadata = [
    {
        "name": "Diagnostics",
        "description": "System health check and database connectivity verification.",
    },
    {
        "name": "Chat",
        "description": "Core RAG chat operations utilizing semantic search and LLM context injection.",
    },
    {
        "name": "Ingestion",
        "description": "High-throughput asynchronous bulk ingestion endpoints.",
    },
]

from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure ingestion collection is initialized on startup
    try:
        await run_sync(ensure_ingest_collection_initialized, _retriever)
        print(f"Ingest collection '{settings.ingest_collection_name}' is initialized.")
    except Exception as e:
        print(f"Warning: Failed to ensure ingest collection on boot: {e}")

    # Set up local asyncio queue and start worker process using asyncio subprocess
    import sys
    import orjson

    # Pin the current Uvicorn worker process to Cores 4, 5, 6, and 7
    try:
        os.sched_setaffinity(0, {4, 5, 6, 7})
        print(f"Uvicorn worker process pinned to cores {os.sched_getaffinity(0)}", flush=True)
    except Exception as e:
        print(f"Warning: Failed to pin Uvicorn worker process: {e}", flush=True)

    app.state.local_queue = asyncio.Queue(maxsize=1200000)
    app.state.image_queue = asyncio.Queue(maxsize=1200000)
    
    # Spawn child worker process on Cores 8 to 11 using taskset
    app.state.ingest_worker_process = await asyncio.create_subprocess_exec(
        "taskset",
        "-c",
        "8-11",
        sys.executable,
        "-u",
        "-m",
        "ingestion.ingest_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=None,
        stderr=None,
    )

    # Spawn child image worker process on Cores 12 to 15 using taskset
    app.state.image_worker_process = await asyncio.create_subprocess_exec(
        "taskset",
        "-c",
        "12-15",
        sys.executable,
        "-u",
        "-m",
        "ingestion.ingest_image_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=None,
        stderr=None,
    )

    # General-purpose background IPC writer task to write enqueued items to the worker's stdin
    async def run_ipc_writer_loop(queue: asyncio.Queue, process, name: str):
        writer = process.stdin
        if not writer:
            print(f"Warning: stdin not available for {name} subprocess.")
            return
        while True:
            try:
                # Block waiting for the first item
                item = await queue.get()
                batch = [item]
                
                # Drain queue up to 1000 items without yielding
                while len(batch) < 1000:
                    try:
                        batch.append(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                
                # Serialize batch using orjson and write to stdin
                payload = b"".join(orjson.dumps(x) + b"\n" for x in batch)
                writer.write(payload)
                await writer.drain()
                
                for _ in range(len(batch)):
                    queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in {name} IPC writer loop: {e}")
                await asyncio.sleep(0.1)

    app.state.ipc_writer_task = asyncio.create_task(
        run_ipc_writer_loop(app.state.local_queue, app.state.ingest_worker_process, "text_worker")
    )
    app.state.image_ipc_writer_task = asyncio.create_task(
        run_ipc_writer_loop(app.state.image_queue, app.state.image_worker_process, "image_worker")
    )


    # Initialize memory buffer for logging and start background flusher
    app.state.log_buffer = []
    async def log_flusher():
        while True:
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            buffer = app.state.log_buffer
            if buffer:
                app.state.log_buffer = []
                try:
                    os.makedirs("processed", exist_ok=True)
                    with open("processed/concurrency_proof.log", "a", encoding="utf-8") as f:
                        f.writelines(buffer)
                except Exception:
                    pass

    app.state.log_flusher_task = asyncio.create_task(log_flusher())

    yield

    # Graceful shutdown: signal worker processes to exit and join
    print("Shutting down ingest and image worker processes...")
    if hasattr(app.state, "ipc_writer_task"):
        app.state.ipc_writer_task.cancel()
        try:
            await app.state.ipc_writer_task
        except asyncio.CancelledError:
            pass

    if hasattr(app.state, "image_ipc_writer_task"):
        app.state.image_ipc_writer_task.cancel()
        try:
            await app.state.image_ipc_writer_task
        except asyncio.CancelledError:
            pass

    # Terminate text worker
    if hasattr(app.state, "ingest_worker_process") and app.state.ingest_worker_process.stdin:
        try:
            # Send shutdown sentinel to worker process
            app.state.ingest_worker_process.stdin.write(orjson.dumps(None) + b"\n")
            await app.state.ingest_worker_process.stdin.drain()
            app.state.ingest_worker_process.stdin.close()
            await app.state.ingest_worker_process.stdin.wait_closed()
        except Exception as e:
            print(f"Error closing worker process stdin: {e}")

    if hasattr(app.state, "ingest_worker_process"):
        try:
            # Wait for process to exit
            await asyncio.wait_for(app.state.ingest_worker_process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            print("Worker process did not exit in time, killing...")
            try:
                app.state.ingest_worker_process.kill()
                await app.state.ingest_worker_process.wait()
            except Exception as e:
                print(f"Error killing worker process: {e}")
        except Exception as e:
            print(f"Error waiting for worker process: {e}")

    # Terminate image worker
    if hasattr(app.state, "image_worker_process") and app.state.image_worker_process.stdin:
        try:
            # Send shutdown sentinel to image worker process
            app.state.image_worker_process.stdin.write(orjson.dumps(None) + b"\n")
            await app.state.image_worker_process.stdin.drain()
            app.state.image_worker_process.stdin.close()
            await app.state.image_worker_process.stdin.wait_closed()
        except Exception as e:
            print(f"Error closing image worker process stdin: {e}")

    if hasattr(app.state, "image_worker_process"):
        try:
            # Wait for process to exit
            await asyncio.wait_for(app.state.image_worker_process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            print("Image worker process did not exit in time, killing...")
            try:
                app.state.image_worker_process.kill()
                await app.state.image_worker_process.wait()
            except Exception as e:
                print(f"Error killing image worker process: {e}")
        except Exception as e:
            print(f"Error waiting for image worker process: {e}")
    
    # Cancel log flusher and run a final write of any remaining buffered logs
    if hasattr(app.state, "log_flusher_task"):
        app.state.log_flusher_task.cancel()
        try:
            await app.state.log_flusher_task
        except asyncio.CancelledError:
            pass
            
    buffer = getattr(app.state, "log_buffer", [])
    if buffer:
        try:
            os.makedirs("processed", exist_ok=True)
            with open("processed/concurrency_proof.log", "a", encoding="utf-8") as f:
                f.writelines(buffer)
        except Exception:
            pass
            
    print("Ingest worker task stopped.")

app = FastAPI(
    title="FastAPI RAG Service",
    summary="A high-performance asynchronous RAG service using Qdrant and LLMs.",
    description="""
This service implements a configuration-driven Retrieval-Augmented Generation (RAG) pipeline:
* **Ingestion/Storage**: Markdown documentation from FastAPI is indexed in Qdrant (using MiniLM or Granite embeddings).
* **Retrieval**: Leverages dense, sparse, and ColBERT late-interaction reranking.
* **Context Reconstruction**: Inline code blocks (`{* ... *}`) are dynamically resolved and injected into the retrieval context.
* **Generation**: Answers are synthesized using generative LLM providers (Gemini or OpenAI).
    """,
    version="0.1.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate Retriever using the global settings
_retriever = Retriever()

def get_retriever() -> Retriever:
    return _retriever

@app.get(
    "/health", 
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": HealthResponse,
            "description": "Service is unhealthy (e.g. database connection failed or active collection not found)."
        }
    },
    summary="Retrieve service health status",
    description="Validates the connection to the active Qdrant database instance and checks if the configured active collection is present.",
    tags=["Diagnostics"]
)
async def health_check(
    response: Response,
    retriever: Annotated[Retriever, Depends(get_retriever)]
) -> HealthResponse:
    """
    Check the health of the service, verifying the Qdrant connection
    and validating that the active collection is accessible.
    
    If the service is unhealthy or degraded, sets the HTTP response status to 503.
    """
    try:
        # Check if Qdrant is responsive in a worker thread
        collection_exists = await run_sync(
            retriever.client.collection_exists, 
            settings.collection_name
        )
        qdrant_connected = True
    except Exception:
        collection_exists = False
        qdrant_connected = False

    if not qdrant_connected:
        status_text = "unhealthy (qdrant database connection failed)"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif not collection_exists:
        status_text = "degraded (active collection not found)"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        status_text = "healthy"

    return HealthResponse(
        status=status_text,
        active_tier=settings.rag_model_tier,
        collection_name=settings.collection_name,
        llm_model=settings.llm_model,
        qdrant_connected=qdrant_connected
    )

@app.post(
    "/chat", 
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "Validation Error in request parameters"
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Internal server/retrieval database failure"
        },
        status.HTTP_502_BAD_GATEWAY: {
            "model": ErrorResponse,
            "description": "LLM API provider communication failure"
        }
    },
    summary="Execute RAG Chat query",
    description="Runs the full RAG pipeline: retrieves matching sections asynchronously from Qdrant, dynamically extracts full markdown sections with source code blocks, and prompts the LLM.",
    tags=["Chat"]
)
async def chat_endpoint(
    request: Annotated[
        ChatRequest,
        Body(
            description="The query and parameters for the RAG search.",
            examples=[
                {
                    "message": "how to implement CORS?",
                    "mode": "hybrid_rerank"
                }
            ]
        )
    ],
    retriever: Annotated[Retriever, Depends(get_retriever)]
) -> ChatResponse:
    """
    Executes the full RAG pipeline:
    1. Retrieves relevant context from Qdrant using the specified mode (dense, sparse, hybrid, hybrid_rerank).
    2. Reconstructs markdown sections and resolves/injects source code.
    3. Prompts the LLM (Gemini or OpenAI) with the context and query.
    """
    # 1. Retrieve candidates & reconstruct context (track latency)
    start_retrieval = time.perf_counter()
    try:
        retrieved_chunks = await run_sync(
            retriever.search,
            request.message,
            request.mode,
            3  # retrieve top 3 results
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Qdrant retrieval error: {e}"
        )

    # Reconstruct context with code injection
    try:
        context = await reconstruct_context(retrieved_chunks)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context reconstruction error: {e}"
        )
    retrieval_latency = time.perf_counter() - start_retrieval

    # 2. Call LLM (track latency & token usage)
    start_generation = time.perf_counter()
    try:
        llm_response, reasoning, token_info = await generate_llm_response(
            request.message,
            context,
            thinking=request.thinking
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API error: {e}"
        )
    generation_latency = time.perf_counter() - start_generation

    # Format document metadata for references
    retrieved_docs_meta = []
    seen = set()
    for chunk in retrieved_chunks:
        payload = chunk.payload
        url = payload.get("section_url", "")
        # De-duplicate references by URL
        if url and url not in seen:
            seen.add(url)
            retrieved_docs_meta.append(
                RetrievedDocMetadata(
                    page_id=payload.get("page_id", ""),
                    heading=payload.get("heading_text", ""),
                    url=url
                )
            )

    # Build response metadata
    metadata = ChatResponseMetadata(
        active_tier=settings.rag_model_tier,
        search_mode=request.mode,
        retrieval_latency_sec=round(retrieval_latency, 4),
        generation_latency_sec=round(generation_latency, 4),
        token_usage=TokenUsage(
            prompt_tokens=token_info.get("prompt_tokens"),
            completion_tokens=token_info.get("completion_tokens"),
            total_tokens=token_info.get("total_tokens")
        )
    )

    return ChatResponse(
        response=llm_response,
        reasoning=reasoning,
        retrieved_documents=retrieved_docs_meta,
        metadata=metadata
    )


@app.post(
    "/api/v1/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "Validation Error in request items"
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Queue full or worker unhealthy"
        }
    },
    summary="Asynchronously ingest document chunks",
    description="Enqueue a list of document chunks for async processing. Returns status 202 Accepted immediately.",
    tags=["Ingestion"]
)
async def ingest_endpoint(
    request: IngestRequest,
    fastapi_request: Request
) -> IngestResponse:
    """
    Asynchronously queues items for embedding and Qdrant upserts.
    Calculates stable identifiers and enqueues items into the in-memory queue.
    """
    start_time = time.perf_counter()
    task_id = str(uuid.uuid4())
    queue = fastapi_request.app.state.local_queue

    # Check if there is enough space in the queue
    if queue.qsize() + len(request.items) > 1200000:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion queue is full. Please try again later."
        )

    for item in request.items:
        try:
            queue.put_nowait(item.dict())
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ingestion queue is full. Please try again later."
            )

    from datetime import datetime
    timestamp = datetime.utcnow().isoformat()
    request_latency_ms = (time.perf_counter() - start_time) * 1000
    
    try:
        current_qsize = queue.qsize()
    except Exception:
        current_qsize = -1

    if hasattr(fastapi_request.app.state, "log_buffer"):
        fastapi_request.app.state.log_buffer.append(
            f"[{timestamp}] [API Ingest] Accepted batch of {len(request.items)} items | "
            f"Queue Size: {current_qsize} | "
            f"Enqueuing Latency: {request_latency_ms:.3f}ms\n"
        )

    return IngestResponse(
        status="accepted",
        task_id=task_id,
        queued_count=len(request.items)
    )


@app.post(
    "/api/v1/ingest/image",
    response_model=IngestImageResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "Validation Error in request items"
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Queue full or worker unhealthy"
        }
    },
    summary="Asynchronously ingest image URLs",
    description="Enqueue a list of image items for async downloading and FashionCLIP embedding. Returns status 202 Accepted immediately.",
    tags=["Ingestion"]
)
async def ingest_image_endpoint(
    request: IngestImageRequest,
    fastapi_request: Request
) -> IngestImageResponse:
    """
    Asynchronously queues image items for download, FashionCLIP embedding, and Qdrant upserts.
    """
    start_time = time.perf_counter()
    task_id = str(uuid.uuid4())
    queue = fastapi_request.app.state.image_queue

    # Check if there is enough space in the queue
    if queue.qsize() + len(request.items) > 1200000:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image ingestion queue is full. Please try again later."
        )

    for item in request.items:
        try:
            # We serialize image_url to string for safety in JSON serialization
            item_dict = item.dict()
            item_dict["image_url"] = str(item_dict["image_url"])
            queue.put_nowait(item_dict)
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Image ingestion queue is full. Please try again later."
            )

    from datetime import datetime
    timestamp = datetime.utcnow().isoformat()
    request_latency_ms = (time.perf_counter() - start_time) * 1000
    
    try:
        current_qsize = queue.qsize()
    except Exception:
        current_qsize = -1

    if hasattr(fastapi_request.app.state, "log_buffer"):
        fastapi_request.app.state.log_buffer.append(
            f"[{timestamp}] [API Image Ingest] Accepted batch of {len(request.items)} items | "
            f"Queue Size: {current_qsize} | "
            f"Enqueuing Latency: {request_latency_ms:.3f}ms\n"
        )

    return IngestImageResponse(
        status="accepted",
        task_id=task_id,
        queued_count=len(request.items)
    )


# Serve project documentation site if it was built
docs_path = "/app/site"
if os.path.exists(docs_path):
    app.mount("/docs/project", StaticFiles(directory=docs_path, html=True), name="project-docs")
elif os.path.exists("site"):
    app.mount("/docs/project", StaticFiles(directory="site", html=True), name="project-docs")

