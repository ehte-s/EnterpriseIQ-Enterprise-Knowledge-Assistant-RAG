"""Embedding generation providers for EnterpriseIQ."""

import hashlib
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseEmbeddingProvider(ABC):
    """Abstract interface for dense embedding generation."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for a batch of documents."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate dense embedding for a single query."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns embedding vector dimension."""
        pass


class SentenceTransformersEmbedding(BaseEmbeddingProvider):
    """Local offline embedding generation via SentenceTransformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dim = 384

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                self._dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model {self.model_name}: {e}")
                raise EmbeddingError(f"Could not load embedding model: {e}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._load_model()
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"SentenceTransformers embedding failed: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        self._load_model()
        try:
            embedding = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(f"Query embedding failed: {e}") from e

    @property
    def dimension(self) -> int:
        return self._dim


class OpenAIEmbedding(BaseEmbeddingProvider):
    """OpenAI API-compatible dense embeddings."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self._client = None
        self._dim = 1536 if "large" not in model_name else 3072

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            settings = get_settings()
            key = self.api_key or settings.OPENAI_API_KEY
            if not key:
                raise EmbeddingError("OpenAI API key not configured for OpenAIEmbedding.")
            self._client = OpenAI(api_key=key, base_url=settings.OPENAI_BASE_URL)
        return self._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = self._get_client()
        try:
            response = client.embeddings.create(input=texts, model=self.model_name)
            return [data.embedding for data in response.data]
        except Exception as e:
            raise EmbeddingError(f"OpenAI batch embedding failed: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        results = self.embed_documents([text])
        return results[0]

    @property
    def dimension(self) -> int:
        return self._dim


class MockEmbedding(BaseEmbeddingProvider):
    """Deterministic pseudo-semantic embedding for instant testing without external dependencies."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    def _text_to_vec(self, text: str) -> List[float]:
        # Generate a deterministic normalized vector from SHA-256 hash and bag-of-words
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dimension).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._text_to_vec(text)

    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingFactory:
    """Factory to instantiate the appropriate embedding provider."""

    @staticmethod
    def get_provider(
        provider_type: Optional[str] = None,
        model_name: Optional[str] = None,
        mock: bool = False
    ) -> BaseEmbeddingProvider:
        if mock:
            return MockEmbedding()

        settings = get_settings()
        prov = (provider_type or settings.EMBEDDING_PROVIDER).lower()
        model = model_name or settings.EMBEDDING_MODEL

        if prov in ("sentence-transformers", "local", "hf"):
            return SentenceTransformersEmbedding(model_name=model)
        elif prov in ("openai", "remote"):
            return OpenAIEmbedding(model_name=model)
        elif prov in ("mock", "test"):
            return MockEmbedding(dimension=settings.EMBEDDING_DIMENSION)
        else:
            logger.warning(f"Unknown embedding provider '{prov}', falling back to MockEmbedding.")
            return MockEmbedding(dimension=settings.EMBEDDING_DIMENSION)
