"""Unit tests for retrieval components: dense search, BM25, hybrid fusion, and reranking."""

import pytest
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.keyword_search import BM25Searcher
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import QdrantVectorStore


@pytest.fixture
def populated_retriever(in_memory_vector_store, mock_embedding, sample_text_document, sample_markdown_document):
    """Provides a retriever pre-populated with sample documents."""
    pipeline = IngestionPipeline(
        vector_store=in_memory_vector_store,
        embedding_provider=mock_embedding,
    )
    pipeline.process_file(
        sample_text_document.encode("utf-8"),
        filename="acme_hr_policy.txt",
        document_id="doc_hr",
    )
    pipeline.process_file(
        sample_markdown_document.encode("utf-8"),
        filename="acme_it_security.md",
        document_id="doc_sec",
    )

    bm25 = BM25Searcher()
    retriever = HybridRetriever(
        vector_store=in_memory_vector_store,
        embedding_provider=mock_embedding,
        bm25_searcher=bm25,
    )
    retriever.sync_bm25_index()
    return retriever


def test_bm25_search_keyword_matching(populated_retriever):
    """Verify BM25 retrieves chunks with exact keyword matches."""
    results = populated_retriever.retrieve_sparse("parental leave caregiver", top_k=3)
    assert len(results) > 0
    assert any("parental" in r["text"].lower() or "caregiver" in r["text"].lower() for r in results)
    assert results[0]["document_id"] == "doc_hr"


def test_hybrid_search_rrf(populated_retriever):
    """Verify Reciprocal Rank Fusion combines dense and sparse candidate sets."""
    results = populated_retriever.retrieve_hybrid(
        query="password requirements rotation days",
        top_k=3,
        fusion_method="rrf"
    )
    assert len(results) > 0
    # Top result should be the security policy chunk mentioning password
    top_chunk = results[0]
    assert "password" in top_chunk["text"].lower()
    assert top_chunk["document_id"] == "doc_sec"
    assert top_chunk["fusion_method"] == "rrf"


def test_hybrid_search_weighted(populated_retriever):
    """Verify weighted score fusion produces scores in valid ranges."""
    results = populated_retriever.retrieve_hybrid(
        query="annual vacation leave days",
        top_k=3,
        fusion_method="weighted",
        dense_weight=0.7
    )
    assert len(results) > 0
    assert results[0]["fusion_method"] == "weighted"
    assert 0.0 <= results[0]["score"] <= 1.0


def test_cross_encoder_reranker():
    """Verify reranker reorders and normalizes scores."""
    reranker = CrossEncoderReranker(use_mock=True)

    candidates = [
        {"chunk_id": "c1", "text": "Something completely unrelated to cats and dogs.", "score": 0.8},
        {"chunk_id": "c2", "text": "Annual leave policy states employees get 25 days vacation.", "score": 0.5},
    ]

    reranked = reranker.rerank("How many annual leave vacation days?", candidates, top_k=2)

    assert len(reranked) == 2
    # c2 contains more matching keywords so it should be reranked to top
    assert reranked[0]["chunk_id"] == "c2"
    assert reranked[0]["score"] > reranked[1]["score"]
