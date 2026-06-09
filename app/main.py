from typing import Annotated
import time
import os
from fastapi import FastAPI, HTTPException, status, Body, Response
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
)

tags_metadata = [
    {
        "name": "Diagnostics",
        "description": "System health check and database connectivity verification.",
    },
    {
        "name": "Chat",
        "description": "Core RAG chat operations utilizing semantic search and LLM context injection.",
    },
]

from fastapi.middleware.cors import CORSMiddleware

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
retriever = Retriever()

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
async def health_check(response: Response) -> HealthResponse:
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
    ]
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


# Serve project documentation site if it was built
docs_path = "/app/site"
if os.path.exists(docs_path):
    app.mount("/docs/project", StaticFiles(directory=docs_path, html=True), name="project-docs")
elif os.path.exists("site"):
    app.mount("/docs/project", StaticFiles(directory="site", html=True), name="project-docs")

