"""Pydantic schemas for documents, chunks, and ingestion."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentFileType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class ChunkMetadata(BaseModel):
    """Metadata associated with an individual document chunk."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str = Field(..., description="Parent document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: DocumentFileType = Field(..., description="Document type")
    page_number: Optional[int] = Field(None, description="Page number where chunk originated")
    section_heading: Optional[str] = Field(None, description="Section heading if detected")
    chunk_index: int = Field(..., description="Index position of chunk in document")
    content_hash: str = Field(..., description="SHA-256 hash of the parent document")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentChunk(BaseModel):
    """Individual chunk of text with metadata."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class DocumentMetadata(BaseModel):
    """Metadata for an indexed document."""
    document_id: str = Field(..., description="Unique document ID (e.g. doc_abc123)")
    filename: str
    file_type: DocumentFileType
    file_size_bytes: int
    content_hash: str
    chunk_count: int = 0
    total_tokens_est: int = 0
    upload_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """Response returned upon document upload and indexing."""
    document_id: str
    filename: str
    file_type: DocumentFileType
    chunk_count: int
    content_hash: str
    status: str = "indexed"
    message: str


class DocumentListResponse(BaseModel):
    """Response returned when listing documents."""
    total_documents: int
    documents: List[DocumentMetadata]
