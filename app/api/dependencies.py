"""FastAPI dependency injection providers."""

from functools import lru_cache
from typing import Optional
from app.cache.semantic_cache import BaseSemanticCache, SemanticCacheFactory
from app.core.config import Settings, get_settings
from app.generation.answer_generator import AnswerGeneratorFactory
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.embeddings import EmbeddingFactory
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.keyword_search import BM25Searcher
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import QdrantVectorStore
from app.services.document_service import DocumentService
from app.services.query_service import QueryService

# Global service singletons
_vector_store: Optional[QdrantVectorStore] = None
_embedding_provider = None
_hybrid_retriever: Optional[HybridRetriever] = None
_semantic_cache: Optional[BaseSemanticCache] = None
_document_service: Optional[DocumentService] = None
_query_service: Optional[QueryService] = None


def get_semantic_cache() -> BaseSemanticCache:
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCacheFactory.get_cache()
    return _semantic_cache


def get_vector_store() -> QdrantVectorStore:
    global _vector_store, _embedding_provider
    if _vector_store is None:
        _embedding_provider = EmbeddingFactory.get_provider()
        _vector_store = QdrantVectorStore(dimension=_embedding_provider.dimension)
    return _vector_store


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever, _vector_store, _embedding_provider
    if _hybrid_retriever is None:
        vs = get_vector_store()
        bm25 = BM25Searcher()
        _hybrid_retriever = HybridRetriever(
            vector_store=vs,
            embedding_provider=_embedding_provider,
            bm25_searcher=bm25,
        )
        _hybrid_retriever.sync_bm25_index()
    return _hybrid_retriever


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        vs = get_vector_store()
        retriever = get_hybrid_retriever()
        cache = get_semantic_cache()
        pipeline = IngestionPipeline(
            vector_store=vs,
            embedding_provider=_embedding_provider,
        )
        _document_service = DocumentService(pipeline=pipeline, vector_store=vs, semantic_cache=cache)
    return _document_service


def get_query_service() -> QueryService:
    global _query_service
    if _query_service is None:
        retriever = get_hybrid_retriever()
        reranker = CrossEncoderReranker()
        answer_gen = AnswerGeneratorFactory.get_generator()
        cache = get_semantic_cache()
        _query_service = QueryService(
            retriever=retriever,
            reranker=reranker,
            answer_generator=answer_gen,
            semantic_cache=cache,
        )
    return _query_service
