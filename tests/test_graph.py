"""Unit and integration tests for LangGraph state machine execution, conditional edges, and fallbacks."""

import pytest
from app.generation.answer_generator import MockAnswerGenerator
from app.generation.citation_validator import CitationValidator
from app.graph.state import EnterpriseRAGState
from app.graph.workflow import build_rag_graph
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.keyword_search import BM25Searcher
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import QdrantVectorStore
from app.schemas.query import TokenUsage


@pytest.fixture
def graph_test_environment(in_memory_vector_store, mock_embedding, sample_text_document):
    """Sets up a compiled LangGraph with indexed HR policy."""
    pipeline = IngestionPipeline(
        vector_store=in_memory_vector_store,
        embedding_provider=mock_embedding,
    )
    pipeline.process_file(
        sample_text_document.encode("utf-8"),
        filename="leave_policy.txt",
        document_id="doc_hr",
    )

    bm25 = BM25Searcher()
    retriever = HybridRetriever(
        vector_store=in_memory_vector_store,
        embedding_provider=mock_embedding,
        bm25_searcher=bm25,
    )
    retriever.sync_bm25_index()

    reranker = CrossEncoderReranker(use_mock=True)
    answer_generator = MockAnswerGenerator()

    graph = build_rag_graph(
        retriever=retriever,
        reranker=reranker,
        answer_generator=answer_generator,
        relevance_threshold=0.40,
    )
    return graph


def test_successful_rag_workflow(graph_test_environment):
    """Verify that a relevant question executes the complete success path."""
    initial_state: EnterpriseRAGState = {
        "query": "How many days of annual vacation leave do employees get?",
        "conversation_id": "test_conv_001",
        "retrieval_strategy": "hybrid",
        "top_k": 3,
        "use_reranker": True,
        "filters": None,
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

    result = graph_test_environment.invoke(initial_state)

    assert result["status"] == "success"
    assert result["final_answer"] is not None
    assert "annual" in result["final_answer"].lower() or "leave" in result["final_answer"].lower()
    assert result["confidence"] > 0.0
    assert len(result["citations"]) > 0
    assert result["citations"][0].is_valid is True

    # Verify execution path
    expected_nodes = [
        "query_analysis",
        "retrieval",
        "reranking",
        "relevance_check",
        "context_builder",
        "answer_generation",
        "citation_validation",
        "response_validation",
    ]
    for node in expected_nodes:
        assert node in result["graph_path"]


def test_fallback_when_evidence_is_insufficient(mock_embedding):
    """Verify that an unindexed question hits the retry loop and terminates in fallback."""
    # Empty vector store
    empty_store = QdrantVectorStore(collection_name="empty_test", dimension=384, in_memory=True)
    bm25 = BM25Searcher()
    retriever = HybridRetriever(vector_store=empty_store, embedding_provider=mock_embedding, bm25_searcher=bm25)
    reranker = CrossEncoderReranker(use_mock=True)
    answer_generator = MockAnswerGenerator()

    # High threshold to guarantee relevance check failure
    graph = build_rag_graph(
        retriever=retriever,
        reranker=reranker,
        answer_generator=answer_generator,
        relevance_threshold=0.90,
    )

    initial_state: EnterpriseRAGState = {
        "query": "What is the corporate pet insurance policy?",
        "conversation_id": "test_conv_fallback",
        "retrieval_strategy": "hybrid",
        "top_k": 3,
        "use_reranker": True,
        "filters": None,
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

    result = graph.invoke(initial_state)

    assert result["status"] == "fallback"
    assert "does not contain sufficient information" in result["final_answer"]
    assert result["confidence"] == 0.0
    assert result["retry_count"] >= 2
    assert "query_reformulation" in result["graph_path"]
    assert "fallback" in result["graph_path"]


def test_citation_validator_flags_invalid_citations():
    """Verify that CitationValidator marks invalid chunk IDs as is_valid=False."""
    context = [
        {"document_id": "doc_hr", "chunk_id": "doc_hr_c0001", "text": "25 days annual leave.", "score": 0.9}
    ]
    # Model generates citation to existing chunk and non-existing chunk
    answer_text = "Employees get 25 days [DOC:doc_hr:doc_hr_c0001] and also free cars [DOC:doc_fake:chunk_9999]"

    citations = CitationValidator.extract_citations_from_text(answer_text, context)

    assert len(citations) == 2
    valid_cit = next(c for c in citations if c.chunk_id == "doc_hr_c0001")
    invalid_cit = next(c for c in citations if c.chunk_id == "chunk_9999")

    assert valid_cit.is_valid is True
    assert invalid_cit.is_valid is False
