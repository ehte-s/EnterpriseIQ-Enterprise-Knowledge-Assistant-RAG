"""Configuration management for EnterpriseIQ using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """EnterpriseIQ System Settings."""

    # Application
    APP_NAME: str = "EnterpriseIQ"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # LLM Provider Configuration
    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    MODEL_NAME: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.0
    MAX_OUTPUT_TOKENS: int = 1024

    # Embedding Configuration
    EMBEDDING_PROVIDER: str = "sentence-transformers"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Vector Database (Qdrant)
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "enterpriseiq_knowledge"
    QDRANT_PATH: str = "./data/qdrant"

    # Cross-Encoder Reranker
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_N: int = 20
    FINAL_TOP_K: int = 5

    # Semantic Cache
    REDIS_URL: Optional[str] = None
    REDIS_TTL_SECONDS: int = 86400
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = 0.92

    # LangSmith Observability
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "EnterpriseIQ"

    # Ingestion & Chunking Defaults
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50
    MAX_FILE_SIZE_MB: int = 25

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = Field(default_factory=lambda: Path("./data"))

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    return Settings()
