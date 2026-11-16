"""End-to-End evaluator evaluating retrieval, generation faithfulness, citations, and costs."""

import time
from typing import Any, Dict, List, Optional
import numpy as np
from app.core.logging import get_logger
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.generation_metrics import (
    calculate_answer_relevance,
    calculate_citation_correctness,
    calculate_faithfulness,
)
from app.evaluation.metrics import (
    aggregate_retrieval_metrics,
    calculate_hit_rate_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from app.retrieval.embeddings import BaseEmbeddingProvider, EmbeddingFactory
from app.schemas.evaluation import (
    EvaluationQuestion,
    ExperimentResult,
    GenerationMetrics,
    PerformanceMetrics,
    RetrievalMetrics,
)
from app.schemas.query import QueryRequest, RetrievalStrategy
from app.services.query_service import QueryService

logger = get_logger(__name__)


class EndToEndEvaluator:
    """Evaluates the full RAG pipeline across retrieval, generation, citations, and costs."""

    def __init__(
        self,
        query_service: QueryService,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
    ):
        self.query_service = query_service
        self.embedding_provider = embedding_provider or EmbeddingFactory.get_provider()

    def evaluate(
        self,
        dataset: EvaluationDataset,
        experiment_id: str,
        name: str,
        description: str,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        top_k: int = 5,
        use_reranker: bool = True,
        use_cache: bool = False,
    ) -> ExperimentResult:
        """Executes full evaluation across all 100 questions."""
        logger.info(f"Starting E2E evaluation: {name} (Questions: {len(dataset)})")

        query_evals: List[Dict[str, Any]] = []
        retrieval_records: List[Dict[str, Any]] = []
        relevance_scores: List[float] = []
        faithfulness_scores: List[float] = []
        citation_precisions: List[float] = []
        latencies_ms: List[float] = []

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost_usd = 0.0

        for q in dataset.questions:
            is_unanswerable = len(q.relevant_document_ids) == 0

            req = QueryRequest(
                query=q.question,
                retrieval_strategy=retrieval_strategy,
                top_k=top_k,
                use_reranker=use_reranker,
                use_cache=use_cache,
            )

            t0 = time.perf_counter()
            response = self.query_service.execute_query(req)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            # Retrieval metrics (for answerable questions)
            retrieved_doc_ids = [c.document_id for c in response.citations]
            rel_set = set(q.relevant_document_ids)

            hit_rate = calculate_hit_rate_at_k(retrieved_doc_ids, rel_set, top_k) if not is_unanswerable else 1.0
            recall = calculate_recall_at_k(retrieved_doc_ids, rel_set, top_k) if not is_unanswerable else 1.0
            precision = calculate_precision_at_k(retrieved_doc_ids, rel_set, top_k) if not is_unanswerable else 1.0
            mrr = calculate_mrr(retrieved_doc_ids, rel_set) if not is_unanswerable else 1.0
            ndcg = calculate_ndcg_at_k(retrieved_doc_ids, rel_set, top_k) if not is_unanswerable else 1.0

            if not is_unanswerable:
                retrieval_records.append({
                    "hit_rate": hit_rate,
                    "recall": recall,
                    "precision": precision,
                    "mrr": mrr,
                    "ndcg": ndcg,
                })

            # Generation metrics
            rel_score = calculate_answer_relevance(
                generated_answer=response.answer,
                expected_answer=q.expected_answer,
                embedding_provider=self.embedding_provider,
            )
            relevance_scores.append(rel_score)

            # Faithfulness
            context_text = " ".join([c.excerpt for c in response.citations])
            faith_score = calculate_faithfulness(
                generated_answer=response.answer,
                retrieved_context=context_text,
                is_unanswerable=is_unanswerable,
            )
            faithfulness_scores.append(faith_score)

            # Citation correctness
            cit_metrics = calculate_citation_correctness(response.citations, q.relevant_document_ids)
            citation_precisions.append(cit_metrics["citation_precision"])

            # Token telemetry
            token_usage = response.metadata.token_usage
            total_input_tokens += token_usage.prompt_tokens
            total_output_tokens += token_usage.completion_tokens
            total_cost_usd += token_usage.estimated_cost_usd

            query_evals.append({
                "id": q.id,
                "category": q.category,
                "query": q.question,
                "answer": response.answer,
                "status": response.status,
                "confidence": response.confidence,
                "relevance": rel_score,
                "faithfulness": faith_score,
                "citation_metrics": cit_metrics,
                "latency_ms": elapsed_ms,
            })

        # Aggregated Metrics
        avg_retrieval = aggregate_retrieval_metrics(retrieval_records, top_k)
        avg_relevance = float(np.mean(relevance_scores)) if relevance_scores else 0.0
        avg_faithfulness = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0
        avg_citation_precision = float(np.mean(citation_precisions)) if citation_precisions else 0.0

        gen_metrics = GenerationMetrics(
            answer_relevance=round(avg_relevance, 4),
            faithfulness=round(avg_faithfulness, 4),
            citation_correctness=round(avg_citation_precision, 4),
        )

        n = len(dataset.questions)
        avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

        perf_metrics = PerformanceMetrics(
            average_latency_ms=round(avg_latency, 2),
            p95_latency_ms=round(p95_latency, 2),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            average_tokens_per_query=round((total_input_tokens + total_output_tokens) / n, 2),
            estimated_total_cost_usd=round(total_cost_usd, 6),
            cache_hit_rate=0.0,
        )

        from datetime import datetime, timezone
        return ExperimentResult(
            experiment_id=experiment_id,
            name=name,
            description=description,
            configuration={
                "retrieval_strategy": retrieval_strategy.value,
                "top_k": top_k,
                "use_reranker": use_reranker,
                "use_cache": use_cache,
            },
            retrieval_metrics=avg_retrieval,
            generation_metrics=gen_metrics,
            performance_metrics=perf_metrics,
            total_questions=n,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )
