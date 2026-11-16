"""Pydantic schemas for the evaluation framework."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvaluationQuestion(BaseModel):
    """Ground-truth evaluation benchmark item."""
    id: str = Field(..., description="Unique question identifier")
    category: str = Field(..., description="Category e.g. factual, multi_doc, numerical, etc.")
    question: str
    expected_answer: str
    relevant_document_ids: List[str] = Field(default_factory=list)
    relevant_chunk_ids: Optional[List[str]] = Field(default=None)
    expected_evidence: Optional[str] = None


class RetrievalMetrics(BaseModel):
    """Retrieval quality evaluation metrics."""
    hit_rate_at_k: float = Field(..., description="Proportion of queries with >=1 relevant doc in top K")
    recall_at_k: float = Field(..., description="Average proportion of relevant docs retrieved in top K")
    precision_at_k: float = Field(..., description="Average proportion of top K docs that are relevant")
    mrr: float = Field(..., description="Mean Reciprocal Rank")
    ndcg_at_k: float = Field(..., description="Normalized Discounted Cumulative Gain at K")
    top_k: int


class GenerationMetrics(BaseModel):
    """Generation quality metrics."""
    answer_relevance: float = Field(..., description="Semantic alignment of answer to query")
    faithfulness: float = Field(..., description="Groundedness in context without hallucination")
    citation_correctness: float = Field(..., description="Precision and validity of citations")


class PerformanceMetrics(BaseModel):
    """System latency, token usage, and cost metrics."""
    average_latency_ms: float
    p95_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    average_tokens_per_query: float
    estimated_total_cost_usd: float
    cache_hit_rate: float


class ExperimentResult(BaseModel):
    """Full benchmark evaluation run report."""
    experiment_id: str
    name: str
    description: str
    configuration: Dict[str, Any]
    retrieval_metrics: RetrievalMetrics
    generation_metrics: Optional[GenerationMetrics] = None
    performance_metrics: PerformanceMetrics
    total_questions: int
    evaluated_at: str
