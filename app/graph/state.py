"""LangGraph state schema for EnterpriseIQ."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from app.schemas.citation import Citation
from app.schemas.query import TokenUsage


class EnterpriseRAGState(TypedDict):
    """TypedDict state representing the complete lifecycle of a RAG query execution."""

    # Input parameters
    query: str
    conversation_id: Optional[str]
    retrieval_strategy: str
    top_k: int
    use_reranker: bool
    filters: Optional[Dict[str, Any]]

    # Step 1: Query Analysis & Routing
    query_analysis: Dict[str, Any]
    current_retrieval_query: str

    # Step 2 & 3: Retrieval & Reranking
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]

    # Step 4: Relevance Evaluation & Loop Control
    relevance_score: float
    is_relevant: bool
    retry_count: int
    max_retries: int

    # Step 5: Context Assembly
    context_blocks: List[Dict[str, Any]]
    generated_context: str

    # Step 6 & 7: Generation & Citation Validation
    raw_answer: Optional[str]
    final_answer: Optional[str]
    confidence: float
    citations: List[Citation]

    # Execution telemetry & output
    status: str  # "success", "fallback", "error"
    error: Optional[str]
    graph_path: List[str]
    token_usage: TokenUsage
    latency_ms: float
    metadata: Dict[str, Any]
