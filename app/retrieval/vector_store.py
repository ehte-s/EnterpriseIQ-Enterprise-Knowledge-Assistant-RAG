"""Qdrant Vector Database integration for EnterpriseIQ."""

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.schemas.citation import ContextBlock
from app.schemas.document import DocumentChunk

logger = get_logger(__name__)


class QdrantVectorStore:
    """Production vector store implementation using Qdrant with local disk and remote server support."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        dimension: int = 384,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        in_memory: bool = False,
    ):
        settings = get_settings()
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.dimension = dimension
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.path = path or settings.QDRANT_PATH
        self.in_memory = in_memory
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models

            if self.in_memory:
                logger.info("Initializing in-memory Qdrant instance.")
                self._client = QdrantClient(":memory:")
            elif self.url and self.url.strip():
                logger.info(f"Connecting to remote Qdrant at {self.url}")
                self._client = QdrantClient(url=self.url, api_key=self.api_key)
            else:
                qdrant_dir = Path(self.path)
                qdrant_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Initializing local disk-backed Qdrant at {qdrant_dir}")
                self._client = QdrantClient(path=str(qdrant_dir))

            self._ensure_collection()
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise VectorStoreError(f"Qdrant initialization error: {e}") from e

    def _ensure_collection(self):
        """Ensures the collection exists with correct vector size and cosine distance."""
        from qdrant_client.http import models

        try:
            collections = self._client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' (dim={self.dimension})")
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.dimension,
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"Error checking/creating Qdrant collection: {e}")
            raise VectorStoreError(f"Error ensuring collection '{self.collection_name}': {e}") from e

    def upsert_chunks(self, chunks: List[DocumentChunk], batch_size: int = 64) -> int:
        """Upsert a list of document chunks into Qdrant."""
        if not chunks:
            return 0

        from qdrant_client.http import models

        try:
            points: List[models.PointStruct] = []
            for chunk in chunks:
                if not chunk.embedding:
                    raise VectorStoreError(f"Chunk '{chunk.chunk_id}' is missing embedding vector.")

                # Ensure deterministic or valid UUID for Qdrant point ID
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

                payload: Dict[str, Any] = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.metadata.document_id,
                    "filename": chunk.metadata.filename,
                    "file_type": chunk.metadata.file_type.value if hasattr(chunk.metadata.file_type, "value") else str(chunk.metadata.file_type),
                    "page_number": chunk.metadata.page_number,
                    "section_heading": chunk.metadata.section_heading,
                    "chunk_index": chunk.metadata.chunk_index,
                    "content_hash": chunk.metadata.content_hash,
                    "text": chunk.text,
                }

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=chunk.embedding,
                        payload=payload
                    )
                )

            # Batch upsert
            total_points = len(points)
            for i in range(0, total_points, batch_size):
                batch = points[i:i + batch_size]
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )

            logger.info(f"Upserted {total_points} chunks into collection '{self.collection_name}'")
            return total_points
        except Exception as e:
            logger.error(f"Failed to upsert chunks into Qdrant: {e}")
            raise VectorStoreError(f"Upsert error: {e}") from e

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Dense similarity search returning matched chunks with relevance scores."""
        from qdrant_client.http import models

        try:
            query_filter = None
            if filters:
                conditions = []
                for k, v in filters.items():
                    conditions.append(
                        models.FieldCondition(
                            key=k,
                            match=models.MatchValue(value=v)
                        )
                    )
                if conditions:
                    query_filter = models.Filter(must=conditions)

            # Perform query
            if hasattr(self._client, "query_points"):
                # Qdrant client >= 1.9 query_points API
                search_result = self._client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True
                ).points
            else:
                # Older search API fallback
                search_result = self._client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True
                )

            results: List[Dict[str, Any]] = []
            for hit in search_result:
                payload = hit.payload or {}
                results.append({
                    "chunk_id": payload.get("chunk_id", ""),
                    "document_id": payload.get("document_id", ""),
                    "filename": payload.get("filename", ""),
                    "page_number": payload.get("page_number"),
                    "section_heading": payload.get("section_heading"),
                    "chunk_index": payload.get("chunk_index", 0),
                    "text": payload.get("text", ""),
                    "score": float(hit.score),
                })
            return results
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            raise VectorStoreError(f"Vector search failed: {e}") from e

    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks belonging to a document_id."""
        from qdrant_client.http import models

        try:
            filter_selector = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            )
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=filter_selector
            )
            logger.info(f"Deleted all chunks for document_id: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise VectorStoreError(f"Delete document error: {e}") from e

    def count_chunks(self) -> int:
        """Returns total number of chunks in the collection."""
        try:
            info = self._client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception as e:
            logger.warning(f"Could not retrieve point count: {e}")
            return 0

    def scroll_all(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Scroll all chunks to retrieve payloads for BM25 synchronization."""
        try:
            points, _ = self._client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            return [p.payload for p in points if p.payload]
        except Exception as e:
            logger.error(f"Error scrolling chunks: {e}")
            return []
