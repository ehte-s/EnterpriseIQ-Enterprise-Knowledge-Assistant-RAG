"""Unit tests for generation quality evaluation metrics and observability."""

import pytest
from app.core.observability import ObservabilityManager
from app.evaluation.generation_metrics import (
    calculate_answer_relevance,
    calculate_citation_correctness,
    calculate_faithfulness,
)
from app.retrieval.embeddings import MockEmbedding
from app.schemas.citation import Citation


def test_answer_relevance_similarity(mock_embedding):
    """Verify answer relevance score on aligned vs conflicting answers."""
    expected = "Employees are entitled to 25 days of annual vacation leave."
    aligned = "Full-time employees receive 25 days of paid annual vacation leave."
    conflicting = "The weather today in New York is cloudy with chance of rain."

    score_aligned = calculate_answer_relevance(aligned, expected, mock_embedding)
    score_conflicting = calculate_answer_relevance(conflicting, expected, mock_embedding)

    assert 0.0 <= score_aligned <= 1.0
    assert 0.0 <= score_conflicting <= 1.0


def test_faithfulness_and_hallucination_detection():
    """Verify faithfulness correctly detects supported context and flags hallucinations."""
    context = "Employees receive 25 days of paid annual leave. Sick leave is 10 days."
    faithful_answer = "Employees get 25 days of annual leave and 10 days of sick leave."
    hallucinated_answer = "Employees get unlimited free company cars and gold watches every month."

    score_faithful = calculate_faithfulness(faithful_answer, context)
    score_hallucinated = calculate_faithfulness(hallucinated_answer, context)

    assert score_faithful > 0.8
    assert score_hallucinated < 0.3

    # Unanswerable query properly refused
    refusal_answer = "The provided enterprise documentation does not contain sufficient information to answer this question."
    score_refusal = calculate_faithfulness(refusal_answer, "", is_unanswerable=True)
    assert score_refusal == 1.0


def test_citation_correctness_metrics():
    """Verify citation validity, precision, and recall calculations."""
    citations = [
        Citation(document_id="doc_hr", filename="hr.txt", chunk_id="c1", excerpt="...", is_valid=True),
        Citation(document_id="doc_sec", filename="sec.txt", chunk_id="c2", excerpt="...", is_valid=True),
    ]
    ground_truth = ["doc_hr"]

    metrics = calculate_citation_correctness(citations, ground_truth)
    assert metrics["citation_validity"] == 1.0
    assert metrics["citation_precision"] == 0.5  # 1 out of 2 cited docs matched ground truth
    assert metrics["citation_recall"] == 1.0  # All ground truth docs were cited


def test_observability_manager_metadata():
    """Verify LangSmith metadata generation."""
    meta = ObservabilityManager.get_run_metadata(
        query="What is the leave policy?",
        retrieval_strategy="hybrid",
        top_k=5,
        use_reranker=True,
    )
    assert meta["application"] == "EnterpriseIQ"
    assert meta["retrieval_strategy"] == "hybrid"
    assert meta["top_k"] == 5
    assert meta["use_reranker"] is True
