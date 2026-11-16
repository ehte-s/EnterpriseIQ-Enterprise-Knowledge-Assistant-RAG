"""Evaluator engine for running benchmark datasets against retrieval configurations."""

import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from app.core.logging import get_logger
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.metrics import (
    aggregate_retrieval_metrics,
    calculate_hit_rate_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.schemas.evaluation import EvaluationQuestion, PerformanceMetrics, RetrievalMetrics

logger = get_logger(__name__)


class RetrievalEvaluator:
    """Runs systematic retrieval evaluations and generates empirical metric reports."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        top_k: int = 5,
        strategy: str = "hybrid",
        dense_weight: float = 0.6,
        fusion_method: str = "rrf",
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.strategy = strategy.lower()
        self.dense_weight = dense_weight
        self.fusion_method = fusion_method

    def retrieve_for_query(self, query: str) -> List[Dict[str, Any]]:
        """Executes retrieval according to the evaluator's strategy and reranker configuration."""
        if self.strategy == "dense":
            results = self.retriever.retrieve_dense(query, top_k=self.top_k)
        elif self.strategy == "bm25":
            results = self.retriever.retrieve_sparse(query, top_k=self.top_k)
        elif self.strategy == "hybrid":
            fetch_k = self.top_k * 3 if self.reranker else self.top_k
            results = self.retriever.retrieve_hybrid(
                query=query,
                top_k=fetch_k,
                dense_top_k=fetch_k * 2,
                sparse_top_k=fetch_k * 2,
                fusion_method=self.fusion_method,
                dense_weight=self.dense_weight,
            )
        else:
            raise ValueError(f"Unknown retrieval strategy '{self.strategy}'")

        # Apply reranking if configured
        if self.reranker and results:
            results = self.reranker.rerank(query=query, documents=results, top_k=self.top_k)
        else:
            results = results[:self.top_k]

        return results

    def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
        include_unanswerable: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates all questions in the dataset and computes empirical retrieval metrics."""
        questions = dataset.questions if include_unanswerable else dataset.get_answerable_questions()
        logger.info(f"Starting retrieval evaluation across {len(questions)} questions (strategy: {self.strategy})")

        query_evals: List[Dict[str, Any]] = []
        latencies_ms: List[float] = []

        category_evals: Dict[str, List[Dict[str, Any]]] = {}

        for q in questions:
            t0 = time.perf_counter()
            retrieved_chunks = self.retrieve_for_query(q.question)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            # Map retrieved chunks to document IDs
            retrieved_doc_ids = [c.get("document_id", "") for c in retrieved_chunks]
            relevant_doc_ids = set(q.relevant_document_ids)

            hit_rate = calculate_hit_rate_at_k(retrieved_doc_ids, relevant_doc_ids, self.top_k)
            recall = calculate_recall_at_k(retrieved_doc_ids, relevant_doc_ids, self.top_k)
            precision = calculate_precision_at_k(retrieved_doc_ids, relevant_doc_ids, self.top_k)
            mrr = calculate_mrr(retrieved_doc_ids, relevant_doc_ids)
            ndcg = calculate_ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, self.top_k)

            eval_record = {
                "question_id": q.id,
                "category": q.category,
                "hit_rate": hit_rate,
                "recall": recall,
                "precision": precision,
                "mrr": mrr,
                "ndcg": ndcg,
                "latency_ms": elapsed_ms,
                "retrieved_doc_ids": retrieved_doc_ids,
                "relevant_doc_ids": list(relevant_doc_ids),
            }
            query_evals.append(eval_record)

            if q.category not in category_evals:
                category_evals[q.category] = []
            category_evals[q.category].append(eval_record)

        # Aggregate overall metrics
        overall_retrieval_metrics = aggregate_retrieval_metrics(query_evals, self.top_k)

        # Calculate category-level breakdown
        category_breakdown: Dict[str, Dict[str, float]] = {}
        for cat, evals in category_evals.items():
            cat_metrics = aggregate_retrieval_metrics(evals, self.top_k)
            category_breakdown[cat] = {
                "count": len(evals),
                "hit_rate_at_k": cat_metrics.hit_rate_at_k,
                "recall_at_k": cat_metrics.recall_at_k,
                "precision_at_k": cat_metrics.precision_at_k,
                "mrr": cat_metrics.mrr,
            }

        # Latency statistics
        avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

        performance_metrics = PerformanceMetrics(
            average_latency_ms=round(avg_latency, 2),
            p95_latency_ms=round(p95_latency, 2),
            total_input_tokens=0,
            total_output_tokens=0,
            average_tokens_per_query=0.0,
            estimated_total_cost_usd=0.0,
            cache_hit_rate=0.0,
        )

        return {
            "strategy": self.strategy,
            "top_k": self.top_k,
            "has_reranker": self.reranker is not None,
            "total_evaluated_queries": len(query_evals),
            "retrieval_metrics": overall_retrieval_metrics.model_dump(),
            "performance_metrics": performance_metrics.model_dump(),
            "category_breakdown": category_breakdown,
            "query_evaluations": query_evals,
        }
