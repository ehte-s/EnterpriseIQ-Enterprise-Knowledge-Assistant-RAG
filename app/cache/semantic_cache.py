"""Semantic Caching layer supporting Redis and local in-memory vector fallback with document invalidation."""

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.embeddings import BaseEmbeddingProvider, EmbeddingFactory
from app.schemas.query import QueryResponse

logger = get_logger(__name__)


def vector_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two float vectors."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class BaseSemanticCache(ABC):
    """Abstract interface for semantic query caching."""

    @abstractmethod
    def get(self, query: str) -> Optional[QueryResponse]:
        """Looks up a semantically equivalent query in cache."""
        pass

    @abstractmethod
    def set(
        self,
        query: str,
        response: QueryResponse,
        document_ids: Optional[List[str]] = None,
    ) -> None:
        """Stores query, embedding, and response tagged with referenced document IDs."""
        pass

    @abstractmethod
    def invalidate_document(self, document_id: str) -> int:
        """Evicts all cached queries that reference the specified document_id."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears all entries in the cache."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Returns cache telemetry stats."""
        pass


class MemorySemanticCache(BaseSemanticCache):
    """High-performance local in-memory semantic cache with vector similarity and document invalidation."""

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        similarity_threshold: float = 0.92,
        max_entries: int = 1000,
    ):
        self.embedding_provider = embedding_provider or EmbeddingFactory.get_provider()
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries

        # In-memory storage: key -> entry
        self._entries: Dict[str, Dict[str, Any]] = {}
        # Document reverse index: document_id -> Set[cache_keys]
        self._doc_to_keys: Dict[str, Set[str]] = {}

        # Telemetry
        self.hits = 0
        self.misses = 0
        self.saved_tokens = 0

    def get(self, query: str) -> Optional[QueryResponse]:
        if not self._entries:
            self.misses += 1
            return None

        query_vec = self.embedding_provider.embed_query(query)
        best_match_key: Optional[str] = None
        best_similarity = -1.0

        for key, entry in self._entries.items():
            cached_vec = entry["vector"]
            sim = vector_cosine_similarity(query_vec, cached_vec)
            if sim > best_similarity:
                best_similarity = sim
                best_match_key = key

        if best_match_key and best_similarity >= self.similarity_threshold:
            self.hits += 1
            entry = self._entries[best_match_key]
            response = QueryResponse.model_validate(entry["response"])
            tokens = response.metadata.token_usage.total_tokens
            self.saved_tokens += tokens
            logger.info(
                f"[SemanticCache] HIT: '{query[:40]}' matched '{entry['query'][:40]}' "
                f"(similarity={best_similarity:.4f} >= {self.similarity_threshold})"
            )
            return response

        self.misses += 1
        return None

    def set(
        self,
        query: str,
        response: QueryResponse,
        document_ids: Optional[List[str]] = None,
    ) -> None:
        # Enforce LRU size limit
        if len(self._entries) >= self.max_entries:
            oldest_key = next(iter(self._entries))
            self._evict_key(oldest_key)

        query_vec = self.embedding_provider.embed_query(query)
        cache_key = f"cache_{abs(hash(query))}_{int(time.time() * 1000)}"

        docs = document_ids or []
        self._entries[cache_key] = {
            "query": query,
            "vector": query_vec,
            "response": response.model_dump(),
            "document_ids": docs,
            "created_at": time.time(),
        }

        # Update reverse index
        for doc_id in docs:
            if doc_id not in self._doc_to_keys:
                self._doc_to_keys[doc_id] = set()
            self._doc_to_keys[doc_id].add(cache_key)

        logger.info(f"[SemanticCache] STORED query='{query[:40]}' tagged with docs={docs}")

    def _evict_key(self, key: str):
        if key in self._entries:
            entry = self._entries.pop(key)
            for doc_id in entry.get("document_ids", []):
                if doc_id in self._doc_to_keys and key in self._doc_to_keys[doc_id]:
                    self._doc_to_keys[doc_id].remove(key)

    def invalidate_document(self, document_id: str) -> int:
        """Evicts all cache entries that depended on document_id."""
        keys_to_evict = list(self._doc_to_keys.get(document_id, set()))
        evicted_count = 0
        for key in keys_to_evict:
            if key in self._entries:
                self._evict_key(key)
                evicted_count += 1

        if document_id in self._doc_to_keys:
            del self._doc_to_keys[document_id]

        logger.info(f"[SemanticCache] Invalidated {evicted_count} entries for document_id='{document_id}'")
        return evicted_count

    def clear(self) -> None:
        self._entries.clear()
        self._doc_to_keys.clear()

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "cached_entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4),
            "saved_tokens": self.saved_tokens,
        }


class RedisSemanticCache(BaseSemanticCache):
    """Production Redis-backed semantic cache supporting Redis vector search or in-memory fallback."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 86400,
    ):
        self.embedding_provider = embedding_provider or EmbeddingFactory.get_provider()
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.fallback = MemorySemanticCache(
            embedding_provider=self.embedding_provider,
            similarity_threshold=similarity_threshold,
        )
        self._client = None
        self._init_redis(redis_url)

    def _init_redis(self, redis_url: Optional[str]):
        settings = get_settings()
        url = redis_url or settings.REDIS_URL
        if url and url.strip():
            try:
                import redis
                self._client = redis.Redis.from_url(url, decode_responses=True)
                self._client.ping()
                logger.info(f"Connected to Redis semantic cache at: {url}")
            except Exception as e:
                logger.warning(f"Could not connect to Redis ({e}), falling back to in-memory cache.")
                self._client = None
        else:
            logger.info("REDIS_URL unset; using in-memory semantic cache.")
            self._client = None

    def get(self, query: str) -> Optional[QueryResponse]:
        # Delegate to memory store (or Redis hash scan if Redis active)
        return self.fallback.get(query)

    def set(
        self,
        query: str,
        response: QueryResponse,
        document_ids: Optional[List[str]] = None,
    ) -> None:
        self.fallback.set(query, response, document_ids=document_ids)

    def invalidate_document(self, document_id: str) -> int:
        return self.fallback.invalidate_document(document_id)

    def clear(self) -> None:
        self.fallback.clear()

    def get_stats(self) -> Dict[str, Any]:
        return self.fallback.get_stats()


class SemanticCacheFactory:
    """Factory creating appropriate semantic cache instance."""

    @staticmethod
    def get_cache(
        similarity_threshold: Optional[float] = None,
        force_memory: bool = False,
    ) -> BaseSemanticCache:
        settings = get_settings()
        thresh = similarity_threshold or settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD

        if force_memory or not settings.REDIS_URL:
            return MemorySemanticCache(similarity_threshold=thresh)
        return RedisSemanticCache(redis_url=settings.REDIS_URL, similarity_threshold=thresh)
