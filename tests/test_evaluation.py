"""Unit tests for evaluation metrics and benchmark dataset."""

import pytest
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.metrics import (
    aggregate_retrieval_metrics,
    calculate_hit_rate_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
)


def test_metric_calculations():
    """Verify standard metric mathematical formulas."""
    relevant = {"doc_1", "doc_2"}
    retrieved = ["doc_3", "doc_1", "doc_4", "doc_2", "doc_5"]

    # Hit Rate@1 should be 0.0 (doc_3 is not in relevant)
    assert calculate_hit_rate_at_k(retrieved, relevant, k=1) == 0.0
    # Hit Rate@2 should be 1.0 (doc_1 is at rank 2)
    assert calculate_hit_rate_at_k(retrieved, relevant, k=2) == 1.0

    # Recall@2: 1 out of 2 relevant retrieved -> 0.5
    assert calculate_recall_at_k(retrieved, relevant, k=2) == 0.5
    # Recall@4: 2 out of 2 relevant retrieved -> 1.0
    assert calculate_recall_at_k(retrieved, relevant, k=4) == 1.0

    # Precision@2: 1 out of 2 retrieved is relevant -> 0.5
    assert calculate_precision_at_k(retrieved, relevant, k=2) == 0.5
    # Precision@4: 2 out of 4 retrieved are relevant -> 0.5
    assert calculate_precision_at_k(retrieved, relevant, k=4) == 0.5

    # MRR: first relevant is at rank 2 -> 1/2 = 0.5
    assert calculate_mrr(retrieved, relevant) == 0.5

    # NDCG@2 should be > 0.0
    ndcg = calculate_ndcg_at_k(retrieved, relevant, k=2)
    assert 0.0 < ndcg <= 1.0


def test_dataset_loading_and_categories():
    """Verify loading 100 questions from JSONL and checking category integrity."""
    dataset = EvaluationDataset("datasets/evaluation_questions.jsonl")
    assert len(dataset) == 100

    answerable = dataset.get_answerable_questions()
    unanswerable = dataset.get_unanswerable_questions()

    assert len(answerable) == 90
    assert len(unanswerable) == 10

    counts = dataset.get_category_counts()
    assert counts["factual"] == 25
    assert counts["numerical"] == 15
    assert counts["multi_doc"] == 15
    assert counts["contextual"] == 15
    assert counts["distractor"] == 10
    assert counts["unanswerable"] == 10
    assert counts["metadata"] == 10


def test_aggregate_retrieval_metrics():
    """Verify aggregating query evaluation records."""
    records = [
        {"hit_rate": 1.0, "recall": 1.0, "precision": 0.5, "mrr": 1.0, "ndcg": 1.0},
        {"hit_rate": 0.0, "recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0},
    ]
    res = aggregate_retrieval_metrics(records, k=2)
    assert res.hit_rate_at_k == 0.5
    assert res.recall_at_k == 0.5
    assert res.mrr == 0.5
    assert res.top_k == 2
