"""Unit tests for Qdrant vector store and document service."""

import pytest
from app.core.exceptions import DocumentNotFoundError
from app.ingestion.pipeline import IngestionPipeline
from app.schemas.document import ChunkMetadata, DocumentChunk, DocumentFileType
from app.services.document_service import DocumentService


def test_vector_store_upsert_and_search(in_memory_vector_store, mock_embedding):
    """Verify storing chunks and performing dense similarity search."""
    text1 = "ACME Corporation vacation leave entitlement is 25 days annually."
    text2 = "Database credentials must be stored securely in HashiCorp Vault."

    emb1 = mock_embedding.embed_query(text1)
    emb2 = mock_embedding.embed_query(text2)

    chunk1 = DocumentChunk(
        chunk_id="doc_hr_c0001",
        text=text1,
        embedding=emb1,
        metadata=ChunkMetadata(
            chunk_id="doc_hr_c0001",
            document_id="doc_hr",
            filename="hr_policy.txt",
            file_type=DocumentFileType.TXT,
            chunk_index=0,
            content_hash="hash_hr_01"
        )
    )

    chunk2 = DocumentChunk(
        chunk_id="doc_sec_c0001",
        text=text2,
        embedding=emb2,
        metadata=ChunkMetadata(
            chunk_id="doc_sec_c0001",
            document_id="doc_sec",
            filename="security.txt",
            file_type=DocumentFileType.TXT,
            chunk_index=0,
            content_hash="hash_sec_01"
        )
    )

    # Upsert into in-memory Qdrant
    upserted_count = in_memory_vector_store.upsert_chunks([chunk1, chunk2])
    assert upserted_count == 2
    assert in_memory_vector_store.count_chunks() == 2

    # Query with identical vector to text1
    search_results = in_memory_vector_store.search(emb1, top_k=2)
    assert len(search_results) == 2
    # The top result should be chunk1 with highest cosine similarity
    assert search_results[0]["chunk_id"] == "doc_hr_c0001"
    assert search_results[0]["score"] > 0.99


def test_vector_store_deletion(in_memory_vector_store, mock_embedding):
    """Verify deleting a document removes its chunks."""
    text = "Temporary data for deletion."
    emb = mock_embedding.embed_query(text)

    chunk = DocumentChunk(
        chunk_id="doc_temp_c0001",
        text=text,
        embedding=emb,
        metadata=ChunkMetadata(
            chunk_id="doc_temp_c0001",
            document_id="doc_temp",
            filename="temp.txt",
            file_type=DocumentFileType.TXT,
            chunk_index=0,
            content_hash="hash_temp"
        )
    )
    in_memory_vector_store.upsert_chunks([chunk])
    assert in_memory_vector_store.count_chunks() > 0

    # Delete
    deleted = in_memory_vector_store.delete_document("doc_temp")
    assert deleted is True


def test_document_service_deduplication(in_memory_vector_store, mock_embedding, sample_text_document):
    """Verify document service skips duplicate content hashes."""
    pipeline = IngestionPipeline(
        vector_store=in_memory_vector_store,
        embedding_provider=mock_embedding
    )
    service = DocumentService(pipeline=pipeline, vector_store=in_memory_vector_store)

    doc_bytes = sample_text_document.encode("utf-8")

    # Initial upload
    resp1 = service.index_file(doc_bytes, "leave_policy.txt")
    assert resp1.status == "indexed"
    assert resp1.chunk_count > 0

    # Duplicate upload with different filename
    resp2 = service.index_file(doc_bytes, "copy_of_leave_policy.txt")
    assert resp2.status == "skipped_duplicate"
    assert resp2.document_id == resp1.document_id

    # List documents
    docs = service.list_documents()
    assert len(docs) == 1

    # Delete document
    service.delete_document(resp1.document_id)
    assert len(service.list_documents()) == 0

    # Deleting non-existent raises DocumentNotFoundError
    with pytest.raises(DocumentNotFoundError):
        service.delete_document("non_existent_id")
