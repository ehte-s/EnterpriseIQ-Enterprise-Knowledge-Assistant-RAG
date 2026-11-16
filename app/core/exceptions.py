"""Domain-specific exception hierarchy for EnterpriseIQ."""

from typing import Any, Dict, Optional


class EnterpriseIQException(Exception):
    """Base exception for all EnterpriseIQ errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentValidationError(EnterpriseIQException):
    """Raised when an uploaded document fails validation."""
    pass


class DocumentParsingError(EnterpriseIQException):
    """Raised when text cannot be parsed from a document."""
    pass


class DocumentNotFoundError(EnterpriseIQException):
    """Raised when a requested document ID does not exist."""
    pass


class IngestionError(EnterpriseIQException):
    """Raised when an error occurs during document ingestion."""
    pass


class EmbeddingError(EnterpriseIQException):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(EnterpriseIQException):
    """Raised when a vector store operation fails."""
    pass


class RetrievalError(EnterpriseIQException):
    """Raised when document retrieval fails."""
    pass


class RerankingError(EnterpriseIQException):
    """Raised when cross-encoder reranking fails."""
    pass


class LLMGenerationError(EnterpriseIQException):
    """Raised when LLM answer generation fails."""
    pass


class CitationValidationError(EnterpriseIQException):
    """Raised when generated citations fail validation."""
    pass
