"""Hybrid retrieval combining dense vector search and sparse BM25 with RRF and score fusion."""

from typing import Any, Dict, List, Optional
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.retrieval.embeddings import BaseEmbeddingProvider
from app.retrieval.keyword_search import BM25Searcher
from app.retrieval.vector_store import QdrantVectorStore

logger = get_logger(__name__)


class HybridRetriever:
    """Combines Qdrant dense vector search and BM25 sparse keyword search."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_provider: BaseEmbeddingProvider,
        bm25_searcher: Optional[BM25Searcher] = None,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.bm25_searcher = bm25_searcher or BM25Searcher()

    def sync_bm25_index(self):
        """Pulls all indexed chunks from vector store to synchronize the BM25 index."""
        try:
            chunks = self.vector_store.scroll_all(limit=5000)
            self.bm25_searcher.build_index(chunks)
            logger.info(f"Synchronized BM25 index with {len(chunks)} chunks from Qdrant.")
        except Exception as e:
            logger.error(f"Failed to synchronize BM25 index: {e}")
            raise RetrievalError(f"BM25 synchronization error: {e}") from e

    def retrieve_dense(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Dense similarity retrieval via query embedding and Qdrant."""
        query_vec = self.embedding_provider.embed_query(query)
        return self.vector_store.search(query_vector=query_vec, top_k=top_k, filters=filters)

    def retrieve_sparse(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Sparse lexical retrieval via BM25."""
        return self.bm25_searcher.search(query=query, top_k=top_k, filters=filters)

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        dense_top_k: int = 15,
        sparse_top_k: int = 15,
        fusion_method: str = "rrf",
        dense_weight: float = 0.6,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Executes both dense and BM25 searches and fuses them using RRF or weighted fusion."""
        dense_results = self.retrieve_dense(query, top_k=dense_top_k, filters=filters)
        sparse_results = self.retrieve_sparse(query, top_k=sparse_top_k, filters=filters)

        if fusion_method == "rrf":
            return self._reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)
        elif fusion_method == "weighted":
            return self._weighted_score_fusion(
                dense_results, sparse_results, dense_weight=dense_weight, top_k=top_k
            )
        else:
            raise ValueError(f"Unknown fusion method '{fusion_method}'. Use 'rrf' or 'weighted'.")

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 5,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF): RRF_score = sum(1 / (k + rank))."""
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Process dense ranks
        for rank, item in enumerate(dense_results):
            cid = item["chunk_id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + (rank + 1)))

        # Process sparse ranks
        for rank, item in enumerate(sparse_results):
            cid = item["chunk_id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + (rank + 1)))

        # Sort by combined RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        final_results = []
        for cid in sorted_cids[:top_k]:
            item = dict(chunk_map[cid])
            item["score"] = float(rrf_scores[cid])
            item["fusion_method"] = "rrf"
            final_results.append(item)

        return final_results

    @staticmethod
    def _weighted_score_fusion(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        dense_weight: float = 0.6,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Weighted score fusion: Score = alpha * dense + (1 - alpha) * sparse."""
        sparse_weight = 1.0 - dense_weight
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for item in dense_results:
            cid = item["chunk_id"]
            chunk_map[cid] = item
            scores[cid] = scores.get(cid, 0.0) + (dense_weight * item["score"])

        for item in sparse_results:
            cid = item["chunk_id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
            scores[cid] = scores.get(cid, 0.0) + (sparse_weight * item["score"])

        sorted_cids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        final_results = []
        for cid in sorted_cids[:top_k]:
            item = dict(chunk_map[cid])
            item["score"] = float(scores[cid])
            item["fusion_method"] = "weighted"
            final_results.append(item)

        return final_results
