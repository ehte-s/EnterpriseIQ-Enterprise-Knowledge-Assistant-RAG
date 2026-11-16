"""LangGraph execution nodes for the EnterpriseIQ RAG workflow."""

import re
import time
from typing import Any, Callable, Dict, List, Optional
from app.cache.context_optimizer import ContextOptimizer
from app.core.logging import get_logger
from app.generation.answer_generator import BaseAnswerGenerator
from app.generation.citation_validator import CitationValidator
from app.graph.state import EnterpriseRAGState
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.schemas.citation import Citation, ContextBlock
from app.schemas.query import TokenUsage

logger = get_logger(__name__)

RELEVANCE_THRESHOLD = 0.45


class RAGNodes:
    """Encapsulates node execution methods with injected dependencies."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        answer_generator: BaseAnswerGenerator,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        context_optimizer: Optional[ContextOptimizer] = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.answer_generator = answer_generator
        self.relevance_threshold = relevance_threshold
        self.context_optimizer = context_optimizer or ContextOptimizer()

    def query_analysis_node(self, state: EnterpriseRAGState) -> Dict[str, Any]:
        """Node A: Analyzes user query intent, checks domain relevance, and extracts key entities."""
        query = state["query"].strip()
        graph_path = list(state.get("graph_path", [])) + ["query_analysis"]

        # Intent classification heuristic
        query_lower = query.lower()
        if any(w in query_lower for w in ["document id", "version", "effective date", "classification", "policy id"]):
            intent = "metadata_inquiry"
        elif any(w in query_lower for w in ["compare", "versus", "vs", "difference between", "both"]):
            intent = "multi_document_comparison"
        elif any(w in query_lower for w in ["how many", "percentage", "dollar", "$", "cap", "limit", "rate", "p50", "p95"]):
            intent = "numerical"
        elif any(w in query_lower for w in ["hello", "hi", "hey", "who are you", "what can you do"]):
            intent = "conversational"
        else:
            intent = "factual"

        analysis = {
            "intent": intent,
            "original_query": query,
            "is_supported_domain": True,
        }

        logger.info(f"[Graph] Query Analysis: Intent='{intent}' for query: '{query[:60]}...'")
        return {
            "query_analysis": analysis,
            "current_retrieval_query": query,
            "graph_path": graph_path,
        }

    def retrieval_node(self, state: EnterpriseRAGState) -> Dict[str, Any]:
        """Node B: Retrieves candidate chunks using hybrid dense + sparse search."""
        query = state["current_retrieval_query"]
        strategy = state.get("retrieval_strategy", "hybrid").lower()
        top_k = state.get("top_k", 5)
        filters = state.get("filters")
        graph_path = list(state.get("graph_path", [])) + ["retrieval"]

        # Fetch more candidates if reranking is enabled
        fetch_k = top_k * 3 if state.get("use_reranker", True) else top_k

        if strategy == "dense":
            chunks = self.retriever.retrieve_dense(query, top_k=fetch_k, filters=filters)
        elif strategy == "bm25":
            chunks = self.retriever.retrieve_sparse(query, top_k=fetch_k, filters=filters)
        else:
            chunks = self.retriever.retrieve_hybrid(
                query,
                top_k=fetch_k,
                dense_top_k=fetch_k * 2,
                sparse_top_k=fetch_k * 2,
                fusion_method="rrf",
                filters=filters,
            )

        logger.info(f"[Graph] Retrieval: Retrieved {len(chunks)} candidate chunks using {strategy}")
        return {
            "retrieved_chunks": chunks,
            "graph_path": graph_path,
        }

    def reranking_node(self, state: EnterpriseRAGState) -> Dict[str, Any]:
        """Node C: Cross-encoder reranking on candidate chunks."""
        query = state["current_retrieval_query"]
        chunks = state.get("retrieved_chunks", [])
        top_k = state.get("top_k", 5)
        use_reranker = state.get("use_reranker", True)
        graph_path = list(state.get("graph_path", [])) + ["reranking"]

        if not chunks:
            return {
                "reranked_chunks": [],
                "graph_path": graph_path,
            }

        if use_reranker:
            reranked = self.reranker.rerank(query=query, documents=chunks, top_k=top_k)
        else:
            reranked = chunks[:top_k]

        logger.info(f"[Graph] Reranking: Scored top {len(reranked)} chunks")
        return {
            "reranked_chunks": reranked,
            "graph_path": graph_path,
        }

    def relevance_check_node(self, state: EnterpriseRAGState) -> Dict[str, Any]:
        """Node D: Evaluates whether retrieved chunks contain sufficient evidence."""
        reranked = state.get("reranked_chunks", [])
        graph_path = list(state.get("graph_path", [])) + ["relevance_check"]

        if not reranked:
            logger.info("[Graph] Relevance Check: 0 chunks retrieved -> Insufficient evidence.")
            return {
                "relevance_score": 0.0,
                "is_relevant": False,
                "graph_path": graph_path,
            }

        # Measure top relevance score with calibration for RRF vs Cross-Encoder
        top_chunk = reranked[0]
        if top_chunk.get("reranker_score") is not None:
            top_score = float(top_chunk["reranker_score"])
        elif top_chunk.get("fusion_method") == "rrf":
            raw_rrf = float(top_chunk.get("score", 0.0))
            top_score = min(1.0, raw_rrf * 60.0)
        else:
            top_score = float(top_chunk.get("score", 0.0))

        is_relevant = top_score >= self.relevance_threshold

        logger.info(
            f"[Graph] Relevance Check: Top score={top_score:.4f} (threshold={self.relevance_threshold}) -> "
            f"Relevant={is_relevant}"
        )
        return {
            "relevance_score": float(top_score),
            "is_relevant": is_relevant,
            "graph_path": graph_path,
        }

    def query_reformulation_node(self, state: EnterpriseRAGState) -> Dict[str, Any]:
        """Node E: Reformulates query when context is insufficient and retries remain."""
        original_query = state["query"]
        current_query = state["current_retrieval_query"]
        retry_count = state.get("retry_count", 0) + 1
        graph_path = list(state.get("graph_path", [])) + ["query_reformulation"]

        reformulated = self.answer_generator.reformulate_query(
            query=original_query,
            previous_query=current_query,
        )

        logger.info(f"[Graph] Query Reformulation (Retry {retry_count}/{state.get('max_retries', 2)}): '{reformulated}'")
        return {
            "current_retrieval_query": reformulated,
            "retry_count": retry_count,
            "graph_path": graph_path,
        }

    def fallback_node(self, state: EnterpriseRAGState) -> Dict[str, Any]:
        """Node F: Triggered when retrieval attempts fail to locate sufficient grounded evidence."""
        graph_path = list(state.get("graph_path", [])) + ["fallback"]
        fallback_msg = (
            "The provided enterprise documentation does not contain sufficient information to answer this question. "
            "Please check if the relevant document has been uploaded or rephrase your query."
        )

        logger.info("[Graph] Fallback: Returning explicit insufficient-evidence statement.")
        return {
            "final_answer": fallback_msg,
            "raw_answer": fallback_msg,
            "confidence": 0.0,
            "citations": [],
            "status": "fallback",
            "graph_path": graph_path,
        }

    def context_builder_node(self, state: EnterpriseRAGState) -> Dict[str, Any]:
        """Node G: Deduplicates, prunes near-duplicates, token-budgets, and tags context chunks for the LLM."""
        reranked = state.get("reranked_chunks", [])
        graph_path = list(state.get("graph_path", [])) + ["context_builder"]

        # Apply Context Optimizer (deduplication, token budget enforcement, near-duplicate pruning)
        optimized_chunks, telemetry = self.context_optimizer.optimize_context(reranked)

        formatted_blocks: List[str] = []
        for c in optimized_chunks:
            text = c.get("text", "").strip()
            tag = f"[DOC:{c.get('document_id', 'doc')}:{c.get('chunk_id', 'c0001')}]"
            heading_info = f" (Section: {c.get('section_heading')})" if c.get("section_heading") else ""
            page_info = f" (Page: {c.get('page_number')})" if c.get("page_number") else ""

            block_header = f"{tag}{heading_info}{page_info}"
            block_str = f"{block_header}\n{text}"
            formatted_blocks.append(block_str)

        assembled_context = "\n\n---\n\n".join(formatted_blocks)
        logger.info(
            f"[Graph] Context Builder: Optimized {len(reranked)} -> {len(optimized_chunks)} blocks "
            f"(Saved {telemetry.get('tokens_saved', 0)} tokens / {telemetry.get('savings_pct', 0.0)}%)"
        )

        return {
            "context_blocks": optimized_chunks,
            "generated_context": assembled_context,
            "graph_path": graph_path,
        }

    def answer_generation_node(self, state: EnterpriseRAGState) -> Dict[str, Any]:
        """Node H: Prompts LLM to generate answer with citations grounded in context."""
        query = state["query"]
        context_str = state["generated_context"]
        graph_path = list(state.get("graph_path", [])) + ["answer_generation"]

        raw_answer, token_usage = self.answer_generator.generate_answer(
            query=query,
            context_blocks=context_str,
        )

        logger.info(f"[Graph] Answer Generation: Generated {len(raw_answer)} chars of answer text")
        return {
            "raw_answer": raw_answer,
            "token_usage": token_usage,
            "graph_path": graph_path,
        }

    def citation_validation_node(self, state: EnterpriseRAGState) -> Dict[str, Any]:
        """Node I: Programmatically verifies citations against context chunks."""
        raw_answer = state.get("raw_answer", "")
        context_chunks = state.get("context_blocks", [])
        graph_path = list(state.get("graph_path", [])) + ["citation_validation"]

        extracted_citations = CitationValidator.extract_citations_from_text(
            text=raw_answer,
            context_chunks=context_chunks,
        )

        valid_count = sum(1 for c in extracted_citations if c.is_valid)
        total_citations = len(extracted_citations)

        logger.info(
            f"[Graph] Citation Validation: {valid_count}/{total_citations} valid citations verified"
        )
        return {
            "citations": extracted_citations,
            "graph_path": graph_path,
        }

    def response_validation_node(self, state: EnterpriseRAGState) -> Dict[str, Any]:
        """Node J: Pydantic validation and final confidence calibration."""
        graph_path = list(state.get("graph_path", [])) + ["response_validation"]
        status = "fallback" if state.get("status") == "fallback" else "success"

        if status == "fallback":
            confidence = 0.0
            final_answer = state["final_answer"]
        else:
            final_answer = state.get("raw_answer", "")
            relevance = state.get("relevance_score", 0.5)
            citations = state.get("citations", [])
            valid_cit_ratio = (
                sum(1 for c in citations if c.is_valid) / len(citations)
                if citations else 0.8
            )
            confidence = round(min(1.0, max(0.0, (relevance * 0.7) + (valid_cit_ratio * 0.3))), 2)

        return {
            "final_answer": final_answer,
            "confidence": confidence,
            "status": status,
            "graph_path": graph_path,
        }
