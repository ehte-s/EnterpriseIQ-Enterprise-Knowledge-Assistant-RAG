"""Pydantic schemas for query requests, responses, and execution metadata."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.citation import Citation


class RetrievalStrategy(str, Enum):
    """Retrieval strategies supported."""
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class QueryRequest(BaseModel):
    """Incoming user query request."""
    query: str = Field(..., min_length=2, max_length=2000, description="User's natural language query")
    conversation_id: Optional[str] = Field(None, description="Optional conversation session ID")
    retrieval_strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.HYBRID,
        description="Retrieval strategy to apply (dense, bm25, or hybrid)"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Final number of context chunks to use")
    use_reranker: bool = Field(default=True, description="Whether to apply cross-encoder reranker")
    use_cache: bool = Field(default=True, description="Whether to check and populate semantic cache")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters e.g. {'document_id': '...'}")


class TokenUsage(BaseModel):
    """Token consumption breakdown."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class QueryExecutionMetadata(BaseModel):
    """Execution telemetry and graph path metadata."""
    retrieval_strategy: str
    retrieved_chunks_count: int
    reranked_chunks_count: int
    relevance_score: float
    retry_count: int
    graph_path: List[str] = Field(default_factory=list)
    cache_hit: bool = False
    latency_ms: float
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class QueryResponse(BaseModel):
    """Pydantic-validated EnterpriseIQ query response."""
    query: str
    answer: str
    status: str = Field("success", description="Status e.g. success, fallback, or error")
    confidence: float = Field(..., ge=0.0, le=1.0, description="System confidence score")
    citations: List[Citation] = Field(default_factory=list)
    metadata: QueryExecutionMetadata
