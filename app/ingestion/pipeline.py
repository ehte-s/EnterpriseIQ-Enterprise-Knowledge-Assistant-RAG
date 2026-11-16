"""Document ingestion pipeline orchestrating parsing, cleaning, chunking, embedding, and vector storage."""

import os
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Union
from app.core.config import get_settings
from app.core.exceptions import DocumentValidationError, IngestionError
from app.core.logging import get_logger
from app.ingestion.chunkers import BaseChunker, ChunkerFactory
from app.ingestion.loaders import DocumentLoaderFactory, LoadedPage
from app.ingestion.parsers import TextCleaner
from app.retrieval.embeddings import BaseEmbeddingProvider, EmbeddingFactory
from app.retrieval.vector_store import QdrantVectorStore
from app.schemas.document import (
    ChunkMetadata,
    DocumentChunk,
    DocumentFileType,
    DocumentMetadata,
)

logger = get_logger(__name__)


class IngestionPipeline:
    """End-to-end pipeline for ingesting, processing, and indexing documents."""

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        default_chunker: Optional[BaseChunker] = None,
    ):
        settings = get_settings()
        self.embedding_provider = embedding_provider or EmbeddingFactory.get_provider()
        self.vector_store = vector_store or QdrantVectorStore(dimension=self.embedding_provider.dimension)
        self.chunker = default_chunker or ChunkerFactory.create(
            strategy="recursive",
            chunk_size=settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
        )

    def process_file(
        self,
        file_bytes: bytes,
        filename: str,
        document_id: Optional[str] = None,
        chunker: Optional[BaseChunker] = None,
        custom_metadata: Optional[dict] = None,
    ) -> Tuple[DocumentMetadata, List[DocumentChunk]]:
        """Processes a single file from raw bytes through the full ingestion pipeline."""
        settings = get_settings()

        # 1. Validation: File size
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise DocumentValidationError(
                f"File '{filename}' exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB "
                f"(current size: {file_size_mb:.2f}MB)"
            )

        # 2. Compute Content Hash
        content_hash = DocumentLoaderFactory.calculate_sha256(file_bytes)
        doc_id = document_id or f"doc_{content_hash[:12]}"

        # 3. Load & Parse Pages
        loader, file_type = DocumentLoaderFactory.get_loader(filename)
        loaded_pages = loader.load(file_bytes, filename)
        if not loaded_pages:
            raise IngestionError(f"No content extracted from file '{filename}'")

        # 4. Text Cleaning and Chunking
        active_chunker = chunker or self.chunker
        chunks: List[DocumentChunk] = []
        chunk_idx = 0
        total_tokens_est = 0

        for page in loaded_pages:
            cleaned_text = TextCleaner.clean_text(page.text)
            if not cleaned_text:
                continue

            raw_chunks = active_chunker.split_text(cleaned_text)
            detected_heading = page.section_heading or TextCleaner.extract_heading(cleaned_text)

            for text_chunk in raw_chunks:
                text_chunk = text_chunk.strip()
                if not text_chunk:
                    continue

                chunk_id = f"{doc_id}_c{chunk_idx:04d}"
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    filename=filename,
                    file_type=file_type,
                    page_number=page.page_number,
                    section_heading=detected_heading,
                    chunk_index=chunk_idx,
                    content_hash=content_hash,
                )
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=text_chunk,
                    metadata=metadata,
                )
                chunks.append(chunk)
                total_tokens_est += len(text_chunk.split())
                chunk_idx += 1

        if not chunks:
            raise IngestionError(f"No valid chunks generated for document '{filename}'")

        # 5. Batch Embedding Generation
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedding_provider.embed_documents(chunk_texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 6. Store Chunks in Qdrant Vector Store
        self.vector_store.upsert_chunks(chunks)

        # 7. Construct DocumentMetadata summary
        doc_meta = DocumentMetadata(
            document_id=doc_id,
            filename=filename,
            file_type=file_type,
            file_size_bytes=len(file_bytes),
            content_hash=content_hash,
            chunk_count=len(chunks),
            total_tokens_est=total_tokens_est,
            custom_metadata=custom_metadata or {},
        )

        logger.info(
            f"Successfully ingested '{filename}' (ID: {doc_id}) into {len(chunks)} chunks "
            f"(~{total_tokens_est} words)"
        )
        return doc_meta, chunks
