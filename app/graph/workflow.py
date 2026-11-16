"""Compiled LangGraph workflow for EnterpriseIQ."""

from typing import Optional
from langgraph.graph import END, START, StateGraph
from app.core.logging import get_logger
from app.generation.answer_generator import AnswerGeneratorFactory, BaseAnswerGenerator
from app.graph.edges import route_after_relevance
from app.graph.nodes import RAGNodes
from app.graph.state import EnterpriseRAGState
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker

logger = get_logger(__name__)


def build_rag_graph(
    retriever: HybridRetriever,
    reranker: Optional[CrossEncoderReranker] = None,
    answer_generator: Optional[BaseAnswerGenerator] = None,
    relevance_threshold: float = 0.45,
):
    """Builds and compiles the deterministic LangGraph state machine for EnterpriseIQ."""
    reranker_instance = reranker or CrossEncoderReranker(use_mock=False)
    generator_instance = answer_generator or AnswerGeneratorFactory.get_generator()

    nodes = RAGNodes(
        retriever=retriever,
        reranker=reranker_instance,
        answer_generator=generator_instance,
        relevance_threshold=relevance_threshold,
    )

    # Initialize StateGraph with typed schema
    workflow = StateGraph(EnterpriseRAGState)

    # Add all 10 nodes
    workflow.add_node("query_analysis", nodes.query_analysis_node)
    workflow.add_node("retrieval", nodes.retrieval_node)
    workflow.add_node("reranking", nodes.reranking_node)
    workflow.add_node("relevance_check", nodes.relevance_check_node)
    workflow.add_node("query_reformulation", nodes.query_reformulation_node)
    workflow.add_node("fallback", nodes.fallback_node)
    workflow.add_node("context_builder", nodes.context_builder_node)
    workflow.add_node("answer_generation", nodes.answer_generation_node)
    workflow.add_node("citation_validation", nodes.citation_validation_node)
    workflow.add_node("response_validation", nodes.response_validation_node)

    # Define execution edges
    workflow.add_edge(START, "query_analysis")
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "reranking")
    workflow.add_edge("reranking", "relevance_check")

    # Conditional branching from relevance_check
    workflow.add_conditional_edges(
        "relevance_check",
        route_after_relevance,
        {
            "context_builder": "context_builder",
            "query_reformulation": "query_reformulation",
            "fallback": "fallback",
        },
    )

    # Retry loop edge
    workflow.add_edge("query_reformulation", "retrieval")

    # Success generation pipeline
    workflow.add_edge("context_builder", "answer_generation")
    workflow.add_edge("answer_generation", "citation_validation")
    workflow.add_edge("citation_validation", "response_validation")

    # Fallback branch to response validation
    workflow.add_edge("fallback", "response_validation")

    # Terminal edge
    workflow.add_edge("response_validation", END)

    app = workflow.compile()
    logger.info("Compiled LangGraph EnterpriseIQ workflow successfully.")
    return app
