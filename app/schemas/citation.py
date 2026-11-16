"""Pydantic models for citations and context blocks."""

from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Accurate source citation grounded in retrieved evidence."""
    document_id: str = Field(..., description="ID of source document")
    filename: str = Field(..., description="Original filename")
    page_number: Optional[int] = Field(None, description="Page number of cited evidence")
    chunk_id: str = Field(..., description="Specific chunk ID cited")
    excerpt: str = Field(..., description="Verbatim supporting text excerpt")
    relevance_score: Optional[float] = Field(None, description="Relevance or reranker score")
    is_valid: bool = Field(True, description="Whether citation was validated against ground context")


class ContextBlock(BaseModel):
    """Context chunk prepared for the LLM prompt."""
    tag: str = Field(..., description="Formatted tag e.g. [DOC:doc_01:chunk_04]")
    document_id: str
    chunk_id: str
    filename: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    text: str
    score: float
    token_count_est: int
