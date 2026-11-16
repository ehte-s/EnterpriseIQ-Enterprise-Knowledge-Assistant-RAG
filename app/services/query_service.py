"""Query Service orchestrating LangGraph execution, metrics, and semantic caching."""

import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional
from app.core.config import get_settings
from app.core.logging import get_logger
from app.generation.answer_generator import AnswerGeneratorFactory, BaseAnswerGenerator
from app.graph.state import EnterpriseRAGState
from app.graph.workflow import build_rag_graph
from app.retrieval.embeddings import BaseEmbeddingProvider, EmbeddingFactory
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.schemas.citation import Citation
from app.schemas.query import (
    QueryExecutionMetadata,
    QueryRequest,
    QueryResponse,
    TokenUsage,
)

logger = get_logger(__name__)


class QueryService:
    """High-level service coordinating query execution, LangGraph state machine, and caching."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        answer_generator: Optional[BaseAnswerGenerator] = None,
        semantic_cache: Optional[Any] = None,
        relevance_threshold: float = 0.45,
    ):
        settings = get_settings()
        self.embedding_provider = EmbeddingFactory.get_provider()
        self.retriever = retriever
        self.reranker = reranker or CrossEncoderReranker(use_mock=False)
        self.answer_generator = answer_generator or AnswerGeneratorFactory.get_generator()
        self.semantic_cache = semantic_cache
        self.relevance_threshold = relevance_threshold

        self._graph = None
        # Telemetry metrics
        self.total_queries = 0
        self.total_latency_ms = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.cache_hits = 0

    def _get_graph(self):
        if self._graph is None:
            if self.retriever is None:
                raise RuntimeError("Retriever must be initialized before building LangGraph workflow.")
            self._graph = build_rag_graph(
                retriever=self.retriever,
                reranker=self.reranker,
                answer_generator=self.answer_generator,
                relevance_threshold=self.relevance_threshold,
            )
        return self._graph

    def set_retriever(self, retriever: HybridRetriever):
        """Allows injecting or updating the active retriever instance."""
        self.retriever = retriever
        self._graph = None

    def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Executes a complete query request through semantic cache and LangGraph."""
        t_start = time.perf_counter()

        # 1. Semantic Cache Lookup
        if request.use_cache and self.semantic_cache:
            try:
                cached_res = self.semantic_cache.get(request.query)
                if cached_res:
                    self.cache_hits += 1
                    self.total_queries += 1
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    cached_res.metadata.cache_hit = True
                    cached_res.metadata.latency_ms = round(elapsed_ms, 2)
                    logger.info(f"Semantic Cache HIT for query: '{request.query[:50]}'")
                    return cached_res
            except Exception as e:
                logger.warning(f"Semantic cache lookup failed: {e}")

        # 2. Build initial LangGraph state
        initial_state: EnterpriseRAGState = {
            "query": request.query,
            "conversation_id": request.conversation_id,
            "retrieval_strategy": request.retrieval_strategy.value,
            "top_k": request.top_k,
            "use_reranker": request.use_reranker,
            "filters": request.filters,
            "query_analysis": {},
            "current_retrieval_query": "",
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "relevance_score": 0.0,
            "is_relevant": False,
            "retry_count": 0,
            "max_retries": 2,
            "context_blocks": [],
            "generated_context": "",
            "raw_answer": None,
            "final_answer": None,
            "confidence": 0.0,
            "citations": [],
            "status": "in_progress",
            "error": None,
            "graph_path": [],
            "token_usage": TokenUsage(),
            "latency_ms": 0.0,
            "metadata": {},
        }

        # 3. Execute LangGraph Workflow
        graph = self._get_graph()
        final_state = graph.invoke(initial_state)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        token_usage = final_state.get("token_usage", TokenUsage())

        # Update metrics
        self.total_queries += 1
        self.total_latency_ms += elapsed_ms
        self.total_input_tokens += token_usage.prompt_tokens
        self.total_output_tokens += token_usage.completion_tokens

        # 4. Construct Pydantic QueryResponse
        metadata = QueryExecutionMetadata(
            retrieval_strategy=request.retrieval_strategy.value,
            retrieved_chunks_count=len(final_state.get("retrieved_chunks", [])),
            reranked_chunks_count=len(final_state.get("reranked_chunks", [])),
            relevance_score=final_state.get("relevance_score", 0.0),
            retry_count=final_state.get("retry_count", 0),
            graph_path=final_state.get("graph_path", []),
            cache_hit=False,
            latency_ms=round(elapsed_ms, 2),
            token_usage=token_usage,
        )

        response = QueryResponse(
            query=request.query,
            answer=final_state.get("final_answer", ""),
            status=final_state.get("status", "success"),
            confidence=final_state.get("confidence", 0.0),
            citations=final_state.get("citations", []),
            metadata=metadata,
        )

        # 5. Populate Semantic Cache (if successful and enabled)
        if request.use_cache and self.semantic_cache and response.status == "success":
            try:
                referenced_doc_ids = list({c.document_id for c in response.citations})
                self.semantic_cache.set(request.query, response, document_ids=referenced_doc_ids)
            except Exception as e:
                logger.warning(f"Failed to populate semantic cache: {e}")

        return response

    async def execute_query_stream(self, request: QueryRequest) -> AsyncIterator[str]:
        """Provides SSE event streaming for real-time frontend UI feedback."""
        yield f"event: status\ndata: {json.dumps({'stage': 'query_analysis', 'message': 'Analyzing query...'})}\n\n"
        time.sleep(0.05)

        yield f"event: status\ndata: {json.dumps({'stage': 'retrieval', 'message': 'Retrieving knowledge base documents...'})}\n\n"
        time.sleep(0.05)

        # Execute full query
        response = self.execute_query(request)

        # Stream answer chunks progressively
        yield f"event: status\ndata: {json.dumps({'stage': 'generating', 'message': 'Generating verified answer...'})}\n\n"
        words = response.answer.split(" ")
        for i in range(0, len(words), 3):
            chunk_text = " ".join(words[i:i + 3]) + " "
            yield f"event: delta\ndata: {json.dumps({'content': chunk_text})}\n\n"

        # Emit final response object with citations and telemetry
        yield f"event: complete\ndata: {response.model_dump_json()}\n\n"

    def get_metrics(self) -> Dict[str, Any]:
        """Returns service-level query performance telemetry."""
        avg_latency = (self.total_latency_ms / self.total_queries) if self.total_queries > 0 else 0.0
        cache_hit_rate = (self.cache_hits / self.total_queries) if self.total_queries > 0 else 0.0

        return {
            "total_queries": self.total_queries,
            "average_latency_ms": round(avg_latency, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(cache_hit_rate, 4),
        }
