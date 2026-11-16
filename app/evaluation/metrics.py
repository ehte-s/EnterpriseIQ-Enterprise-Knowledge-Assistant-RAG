"""Retrieval and generation evaluation metrics implementation."""

import math
from typing import List, Set
from app.schemas.evaluation import RetrievalMetrics


def calculate_hit_rate_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
    """Hit Rate@K: 1.0 if at least one relevant document is in top K, else 0.0."""
    if not relevant_doc_ids:
        return 0.0
    top_k_retrieved = retrieved_doc_ids[:k]
    return 1.0 if any(doc_id in relevant_doc_ids for doc_id in top_k_retrieved) else 0.0


def calculate_recall_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
    """Recall@K: Proportion of total relevant documents retrieved in top K."""
    if not relevant_doc_ids:
        return 0.0
    top_k_retrieved = set(retrieved_doc_ids[:k])
    matched = len(top_k_retrieved.intersection(relevant_doc_ids))
    return float(matched) / float(len(relevant_doc_ids))


def calculate_precision_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
    """Precision@K: Proportion of retrieved top K documents that are relevant."""
    if k <= 0:
        return 0.0
    top_k_retrieved = retrieved_doc_ids[:k]
    matched = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_doc_ids)
    return float(matched) / float(k)


def calculate_mrr(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str]) -> float:
    """Mean Reciprocal Rank: 1 / rank of first relevant document."""
    if not relevant_doc_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / float(rank)
    return 0.0


def calculate_ndcg_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K (binary relevance)."""
    if not relevant_doc_ids or k <= 0:
        return 0.0

    top_k_retrieved = retrieved_doc_ids[:k]
    dcg = 0.0
    for rank, doc_id in enumerate(top_k_retrieved, start=1):
        rel = 1.0 if doc_id in relevant_doc_ids else 0.0
        dcg += rel / math.log2(rank + 1.0)

    # Ideal DCG (all relevant docs ranked at top)
    idcg = 0.0
    ideal_hits = min(len(relevant_doc_ids), k)
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(rank + 1.0)

    return float(dcg / idcg) if idcg > 0 else 0.0


def aggregate_retrieval_metrics(
    query_evaluations: List[dict],
    k: int,
) -> RetrievalMetrics:
    """Aggregates per-query metric records into a standardized RetrievalMetrics model."""
    if not query_evaluations:
        return RetrievalMetrics(
            hit_rate_at_k=0.0,
            recall_at_k=0.0,
            precision_at_k=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
            top_k=k,
        )

    n = len(query_evaluations)
    avg_hit_rate = sum(q["hit_rate"] for q in query_evaluations) / n
    avg_recall = sum(q["recall"] for q in query_evaluations) / n
    avg_precision = sum(q["precision"] for q in query_evaluations) / n
    avg_mrr = sum(q["mrr"] for q in query_evaluations) / n
    avg_ndcg = sum(q["ndcg"] for q in query_evaluations) / n

    return RetrievalMetrics(
        hit_rate_at_k=round(avg_hit_rate, 4),
        recall_at_k=round(avg_recall, 4),
        precision_at_k=round(avg_precision, 4),
        mrr=round(avg_mrr, 4),
        ndcg_at_k=round(avg_ndcg, 4),
        top_k=k,
    )
