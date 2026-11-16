"""Document Service managing document lifecycle, deduplication, indexing, and registry."""

from typing import Any, Dict, List, Optional
from app.core.exceptions import DocumentNotFoundError
from app.core.logging import get_logger
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.vector_store import QdrantVectorStore
from app.schemas.document import DocumentMetadata, DocumentUploadResponse

logger = get_logger(__name__)


class DocumentService:
    """Service layer managing document registry, deduplication, and vector storage."""

    def __init__(
        self,
        pipeline: Optional[IngestionPipeline] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        semantic_cache: Optional[Any] = None,
    ):
        self.pipeline = pipeline or IngestionPipeline()
        self.vector_store = vector_store or self.pipeline.vector_store
        self.semantic_cache = semantic_cache
        # Registry of indexed documents keyed by document_id
        self._registry: Dict[str, DocumentMetadata] = {}
        # Hash index for deduplication: sha256 -> document_id
        self._hash_to_doc_id: Dict[str, str] = {}

    def index_file(
        self,
        file_bytes: bytes,
        filename: str,
        document_id: Optional[str] = None,
        custom_metadata: Optional[dict] = None,
    ) -> DocumentUploadResponse:
        """Indexes an uploaded file, handling SHA-256 content deduplication."""
        from app.ingestion.loaders import DocumentLoaderFactory

        content_hash = DocumentLoaderFactory.calculate_sha256(file_bytes)

        # Check for duplicate document
        if content_hash in self._hash_to_doc_id:
            existing_doc_id = self._hash_to_doc_id[content_hash]
            existing_doc = self._registry.get(existing_doc_id)
            if existing_doc:
                logger.info(f"Duplicate document detected ({filename}). Reusing ID: {existing_doc_id}")
                return DocumentUploadResponse(
                    document_id=existing_doc_id,
                    filename=existing_doc.filename,
                    file_type=existing_doc.file_type,
                    chunk_count=existing_doc.chunk_count,
                    content_hash=content_hash,
                    status="skipped_duplicate",
                    message="Document with identical content is already indexed.",
                )

        doc_meta, chunks = self.pipeline.process_file(
            file_bytes=file_bytes,
            filename=filename,
            document_id=document_id,
            custom_metadata=custom_metadata,
        )

        # Update registries
        self._registry[doc_meta.document_id] = doc_meta
        self._hash_to_doc_id[content_hash] = doc_meta.document_id

        return DocumentUploadResponse(
            document_id=doc_meta.document_id,
            filename=doc_meta.filename,
            file_type=doc_meta.file_type,
            chunk_count=doc_meta.chunk_count,
            content_hash=content_hash,
            status="indexed",
            message=f"Successfully indexed {len(chunks)} chunks.",
        )

    def get_document(self, document_id: str) -> DocumentMetadata:
        """Retrieves document metadata by document ID."""
        if document_id not in self._registry:
            raise DocumentNotFoundError(f"Document with ID '{document_id}' not found.")
        return self._registry[document_id]

    def list_documents(self) -> List[DocumentMetadata]:
        """Lists all registered documents."""
        return list(self._registry.values())

    def delete_document(self, document_id: str) -> bool:
        """Deletes a document from the registry and removes all chunks from Qdrant."""
        if document_id not in self._registry:
            raise DocumentNotFoundError(f"Document with ID '{document_id}' not found.")

        doc = self._registry.pop(document_id)
        if doc.content_hash in self._hash_to_doc_id:
            del self._hash_to_doc_id[doc.content_hash]

        # Delete chunks from Qdrant
        self.vector_store.delete_document(document_id)
        # Invalidate semantic cache for this document
        if self.semantic_cache:
            self.semantic_cache.invalidate_document(document_id)
        logger.info(f"Deleted document '{document_id}' from vector store, registry, and cache.")
        return True
