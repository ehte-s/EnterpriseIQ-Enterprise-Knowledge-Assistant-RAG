"""Cross-Encoder reranking module for EnterpriseIQ."""

from typing import Any, Dict, List, Optional, Tuple
from app.core.config import get_settings
from app.core.exceptions import RerankingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Reranks candidate chunks using a trained Cross-Encoder model."""

    def __init__(self, model_name: Optional[str] = None, use_mock: bool = False):
        settings = get_settings()
        self.model_name = model_name or settings.RERANKER_MODEL
        self.use_mock = use_mock
        self._model = None

    def _load_model(self):
        if self._model is None and not self.use_mock:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder model: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(
                    f"Could not load CrossEncoder '{self.model_name}': {e}. Falling back to score-based reranking."
                )
                self.use_mock = True

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Scores (query, chunk) pairs and returns chunks sorted descending by relevance score."""
        if not documents:
            return []

        if self.use_mock:
            return self._mock_rerank(query, documents, top_k=top_k)

        self._load_model()
        if self.use_mock:
            return self._mock_rerank(query, documents, top_k=top_k)

        try:
            pairs: List[Tuple[str, str]] = []
            for doc in documents:
                text = doc.get("text", "")
                pairs.append((query, text))

            # Predict cross-encoder scores
            scores = self._model.predict(pairs)

            import numpy as np

            # Apply sigmoid if logits
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))

            scored_docs = []
            for doc, score in zip(documents, scores):
                doc_copy = dict(doc)
                normalized_score = float(sigmoid(score))
                doc_copy["reranker_score"] = normalized_score
                doc_copy["original_score"] = doc.get("score", 0.0)
                doc_copy["score"] = normalized_score
                scored_docs.append(doc_copy)

            # Sort descending
            scored_docs.sort(key=lambda d: d["score"], reverse=True)
            return scored_docs[:top_k]
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            raise RerankingError(f"Reranking execution failed: {e}") from e

    def _mock_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Keyword overlap heuristic fallback when CrossEncoder weights are not available."""
        query_words = set(query.lower().split())
        scored = []
        for doc in documents:
            text = doc.get("text", "").lower()
            overlap = sum(1 for w in query_words if w in text)
            sim_score = min(1.0, 0.5 + (0.1 * overlap))
            doc_copy = dict(doc)
            doc_copy["reranker_score"] = sim_score
            doc_copy["score"] = sim_score
            scored.append(doc_copy)

        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:top_k]
