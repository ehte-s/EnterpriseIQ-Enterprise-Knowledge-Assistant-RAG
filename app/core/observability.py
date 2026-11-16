"""LangSmith observability and tracing integration for EnterpriseIQ."""

import os
from typing import Any, Callable, Dict, Optional
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ObservabilityManager:
    """Manages LangSmith tracing setup, tags, and run metadata."""

    @staticmethod
    def setup_langsmith():
        """Configures LangSmith environment variables if enabled."""
        settings = get_settings()

        if settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT or "EnterpriseIQ"
            logger.info(f"LangSmith tracing enabled for project: {settings.LANGCHAIN_PROJECT}")
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            logger.info("LangSmith tracing disabled (missing key or configured false).")

    @staticmethod
    def get_run_metadata(
        query: str,
        retrieval_strategy: str,
        top_k: int,
        use_reranker: bool,
    ) -> Dict[str, Any]:
        """Generates standard tracing metadata attributes."""
        settings = get_settings()
        return {
            "application": settings.APP_NAME,
            "environment": settings.APP_ENV,
            "retrieval_strategy": retrieval_strategy,
            "top_k": top_k,
            "use_reranker": use_reranker,
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.MODEL_NAME,
        }

    @staticmethod
    def is_tracing_active() -> bool:
        """Checks if LangSmith tracing is currently active in the environment."""
        return os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
