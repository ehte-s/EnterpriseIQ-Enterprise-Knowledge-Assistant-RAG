"""Unit tests for semantic caching, context trimming, and document-level cache invalidation."""

import pytest
from app.cache.context_optimizer import ContextOptimizer
from app.cache.semantic_cache import MemorySemanticCache
from app.retrieval.embeddings import MockEmbedding
from app.schemas.citation import Citation
from app.schemas.query import QueryExecutionMetadata, QueryResponse, TokenUsage


@pytest.fixture
def memory_cache(mock_embedding):
    """Provides an isolated MemorySemanticCache."""
    return MemorySemanticCache(embedding_provider=mock_embedding, similarity_threshold=0.88)


def test_context_optimizer_deduplication_and_budget():
    """Verify that ContextOptimizer removes duplicate content and respects token budgets."""
    optimizer = ContextOptimizer(max_context_tokens=100, similarity_threshold=0.7)

    # 3 chunks, two of which are near-identical
    chunks = [
        {"chunk_id": "c1", "text": "Employees receive 25 days paid annual vacation leave per year.", "score": 0.9},
        {"chunk_id": "c2", "text": "Employees receive 25 days paid annual vacation leave per calendar year.", "score": 0.85},
        {"chunk_id": "c3", "text": "MFA is mandatory with hardware YubiKey tokens.", "score": 0.75},
    ]

    optimized, telemetry = optimizer.optimize_context(chunks, token_budget=60)

    # Near-duplicate c2 should be pruned
    chunk_ids = [c["chunk_id"] for c in optimized]
    assert "c1" in chunk_ids
    assert "c2" not in chunk_ids
    assert telemetry["tokens_saved"] > 0
    assert telemetry["savings_pct"] > 0.0


def test_semantic_cache_exact_and_similar_hits(memory_cache):
    """Verify semantic cache hit on identical and semantically close queries."""
    sample_response = QueryResponse(
        query="What is the leave policy?",
        answer="Employees receive 25 days vacation.",
        status="success",
        confidence=0.95,
        citations=[Citation(document_id="doc_hr", filename="hr.txt", chunk_id="c1", excerpt="...", is_valid=True)],
        metadata=QueryExecutionMetadata(
            retrieval_strategy="hybrid",
            retrieved_chunks_count=1,
            reranked_chunks_count=1,
            relevance_score=0.9,
            retry_count=0,
            graph_path=[],
            cache_hit=False,
            latency_ms=10.0,
            token_usage=TokenUsage(total_tokens=150),
        ),
    )

    # Store in cache
    memory_cache.set("What is the leave policy?", sample_response, document_ids=["doc_hr"])

    # 1. Exact query match -> HIT
    hit_resp = memory_cache.get("What is the leave policy?")
    assert hit_resp is not None
    assert hit_resp.answer == sample_response.answer
    assert memory_cache.hits == 1

    # 2. Dissimilar query -> MISS
    miss_resp = memory_cache.get("How do I configure Kafka in AWS?")
    assert miss_resp is None
    assert memory_cache.misses == 1


def test_document_cache_invalidation(memory_cache):
    """Verify that deleting a document invalidates all dependent cached responses."""
    resp_hr = QueryResponse(
        query="What is annual vacation leave?",
        answer="25 days annual leave.",
        status="success",
        confidence=0.95,
        citations=[],
        metadata=QueryExecutionMetadata(
            retrieval_strategy="hybrid",
            retrieved_chunks_count=1,
            reranked_chunks_count=1,
            relevance_score=0.9,
            retry_count=0,
            graph_path=[],
            cache_hit=False,
            latency_ms=10.0,
            token_usage=TokenUsage(),
        ),
    )

    memory_cache.set("What is annual vacation leave?", resp_hr, document_ids=["doc_hr"])

    # Confirm cache is active
    assert memory_cache.get("What is annual vacation leave?") is not None

    # Invalidate document
    evicted = memory_cache.invalidate_document("doc_hr")
    assert evicted == 1

    # Subsequent lookup must MISS to avoid serving stale answers
    assert memory_cache.get("What is annual vacation leave?") is None
