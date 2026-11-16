"""LangGraph edge definitions and conditional routing functions."""

from typing import Literal
from app.core.logging import get_logger
from app.graph.state import EnterpriseRAGState

logger = get_logger(__name__)


def route_after_relevance(
    state: EnterpriseRAGState,
) -> Literal["context_builder", "query_reformulation", "fallback"]:
    """Conditional routing based on relevance score and retry counter."""
    is_relevant = state.get("is_relevant", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if is_relevant:
        logger.info("[Edge] Evidence is sufficient -> Routing to 'context_builder'")
        return "context_builder"
    elif retry_count < max_retries:
        logger.info(
            f"[Edge] Evidence is insufficient (retry {retry_count}/{max_retries}) -> "
            f"Routing to 'query_reformulation'"
        )
        return "query_reformulation"
    else:
        logger.info(
            f"[Edge] Maximum retries reached ({retry_count}/{max_retries}) without sufficient evidence -> "
            f"Routing to 'fallback'"
        )
        return "fallback"
