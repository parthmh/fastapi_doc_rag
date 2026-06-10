from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Active model configuration tier: 'minilm' or 'granite'
    rag_model_tier: Literal["minilm", "granite"] = "granite"
    
    # LLM configurations
    llm_provider: Literal["gemini", "openai"] = "openai"
    llm_model: str = "mistral-small-2506"
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    llm_base_url: str | None = "https://api.mistral.ai/v1"
    
    # Qdrant configuration
    qdrant_url: str = "http://localhost:6333"
    base_collection_name: str = "fastapi_doc_rag"
    base_ingest_collection_name: str = "fastapi_doc_ingest"
    mock_ingest_embeddings: bool = False
    ingest_batch_size: int = 64
    
    # Shared models
    sparse_model_name: str = "qdrant/bm25"
    colbert_model_name: str = "colbert-ir/colbertv2.0"
    
    @property
    def collection_name(self) -> str:
        return f"{self.base_collection_name}_{self.rag_model_tier}"

    @property
    def ingest_collection_name(self) -> str:
        return f"{self.base_ingest_collection_name}_minilm"
    
    @property
    def dense_model_name(self) -> str:
        if self.rag_model_tier == "granite":
            return "ibm-granite/granite-embedding-english-r2"
        return "sentence-transformers/all-MiniLM-L6-v2"
        
    @property
    def dense_vector_size(self) -> int:
        if self.rag_model_tier == "granite":
            return 768
        return 384
        
    @property
    def chunking_enabled(self) -> bool:
        return self.rag_model_tier == "minilm"
        
    @property
    def chunk_size(self) -> int:
        return 220  # Keep the token limit at 220 as requested by the user

settings = Settings()
