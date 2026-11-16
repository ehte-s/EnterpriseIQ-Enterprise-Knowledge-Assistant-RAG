"""Controlled experiments comparing baseline, intermediate, and optimized RAG retrieval configurations."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.core.logging import get_logger
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.evaluator import RetrievalEvaluator
from app.ingestion.chunkers import ChunkerFactory
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.embeddings import BaseEmbeddingProvider, EmbeddingFactory
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.keyword_search import BM25Searcher
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import QdrantVectorStore

logger = get_logger(__name__)


class ExperimentRunner:
    """Orchestrates controlled experiments to measure retrieval optimization impact."""

    DOCUMENTS = [
        ("datasets/sample_documents/acme_hr_policy.txt", "doc_hr", "acme_hr_policy.txt"),
        ("datasets/sample_documents/acme_it_security_policy.md", "doc_sec", "acme_it_security_policy.md"),
        ("datasets/sample_documents/acme_engineering_architecture.md", "doc_eng", "acme_engineering_architecture.md"),
        ("datasets/sample_documents/acme_procurement_policy.txt", "doc_fin", "acme_procurement_policy.txt"),
    ]

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        dataset_path: Optional[str] = None,
    ):
        self.embedding_provider = embedding_provider or EmbeddingFactory.get_provider()
        self.dataset = EvaluationDataset(dataset_path)

    def _setup_pipeline_for_config(
        self,
        collection_name: str,
        chunk_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> HybridRetriever:
        """Indexes sample documents using the specified chunking configuration and returns a synced retriever."""
        vector_store = QdrantVectorStore(
            collection_name=collection_name,
            dimension=self.embedding_provider.dimension,
            in_memory=True,
        )
        chunker = ChunkerFactory.create(
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        pipeline = IngestionPipeline(
            vector_store=vector_store,
            embedding_provider=self.embedding_provider,
            default_chunker=chunker,
        )

        for file_path, doc_id, filename in self.DOCUMENTS:
            p = Path(file_path)
            if not p.exists():
                raise FileNotFoundError(f"Required document not found: {p}")
            with open(p, "rb") as f:
                pipeline.process_file(
                    file_bytes=f.read(),
                    filename=filename,
                    document_id=doc_id,
                    chunker=chunker,
                )

        bm25 = BM25Searcher()
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_provider=self.embedding_provider,
            bm25_searcher=bm25,
        )
        retriever.sync_bm25_index()
        return retriever

    def run_all_experiments(self) -> Dict[str, Any]:
        """Runs Baseline, Intermediate, and Optimized retrieval experiments and returns comparison results."""
        results: Dict[str, Any] = {}

        # -------------------------------------------------------------
        # Experiment 1: Baseline Configuration
        # Fixed-size chunking (1000/100), Dense-only search, top_k=3, No reranker
        # -------------------------------------------------------------
        logger.info("Executing Experiment 1: Baseline (Fixed 1000/100, Dense-only, Top-3)...")
        baseline_retriever = self._setup_pipeline_for_config(
            collection_name="exp1_baseline",
            chunk_strategy="fixed",
            chunk_size=1000,
            chunk_overlap=100,
        )
        baseline_evaluator = RetrievalEvaluator(
            retriever=baseline_retriever,
            reranker=None,
            top_k=3,
            strategy="dense",
        )
        results["baseline"] = {
            "name": "Configuration A (Baseline)",
            "description": "Fixed chunking (1000/100), Dense-only retrieval, top_k=3, no reranking",
            "eval_result": baseline_evaluator.evaluate_dataset(self.dataset),
        }

        # -------------------------------------------------------------
        # Experiment 2: Intermediate Configuration
        # Recursive chunking (800/100), Hybrid search (RRF), top_k=5, No reranker
        # -------------------------------------------------------------
        logger.info("Executing Experiment 2: Intermediate (Recursive 800/100, Hybrid RRF, Top-5)...")
        intermediate_retriever = self._setup_pipeline_for_config(
            collection_name="exp2_intermediate",
            chunk_strategy="recursive",
            chunk_size=800,
            chunk_overlap=100,
        )
        intermediate_evaluator = RetrievalEvaluator(
            retriever=intermediate_retriever,
            reranker=None,
            top_k=5,
            strategy="hybrid",
            fusion_method="rrf",
        )
        results["intermediate"] = {
            "name": "Configuration B (Intermediate)",
            "description": "Recursive chunking (800/100), Hybrid (Dense + BM25, RRF), top_k=5, no reranking",
            "eval_result": intermediate_evaluator.evaluate_dataset(self.dataset),
        }

        # -------------------------------------------------------------
        # Experiment 3: Optimized Configuration
        # Recursive chunking (500/50), Hybrid search (RRF) + Cross-Encoder reranker, top_k=5
        # -------------------------------------------------------------
        logger.info("Executing Experiment 3: Optimized (Recursive 500/50, Hybrid + Cross-Encoder, Top-5)...")
        optimized_retriever = self._setup_pipeline_for_config(
            collection_name="exp3_optimized",
            chunk_strategy="recursive",
            chunk_size=500,
            chunk_overlap=50,
        )
        reranker = CrossEncoderReranker(use_mock=False)
        optimized_evaluator = RetrievalEvaluator(
            retriever=optimized_retriever,
            reranker=reranker,
            top_k=5,
            strategy="hybrid",
            fusion_method="rrf",
        )
        results["optimized"] = {
            "name": "Configuration C (Optimized)",
            "description": "Recursive chunking (500/50), Hybrid (Dense + BM25, RRF), Cross-Encoder reranker, top_k=5",
            "eval_result": optimized_evaluator.evaluate_dataset(self.dataset),
        }

        return results

    @staticmethod
    def generate_markdown_comparison(results: Dict[str, Any]) -> str:
        """Formats experiment comparison into a clean Markdown table."""
        lines = [
            "# Retrieval Optimization Experiment Results",
            "",
            "| Configuration | Chunking Strategy | Retrieval Strategy | Top-K | Reranker | Hit Rate@K | Recall@K | Precision@K | MRR | Latency (avg) |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for key, exp in results.items():
            metrics = exp["eval_result"]["retrieval_metrics"]
            perf = exp["eval_result"]["performance_metrics"]
            top_k = exp["eval_result"]["top_k"]
            rerank = "Yes (Cross-Encoder)" if exp["eval_result"]["has_reranker"] else "No"
            strat = exp["eval_result"]["strategy"].title()

            if "baseline" in key:
                chunk_desc = "Fixed (1000/100)"
            elif "intermediate" in key:
                chunk_desc = "Recursive (800/100)"
            else:
                chunk_desc = "Recursive (500/50)"

            lines.append(
                f"| {exp['name']} | {chunk_desc} | {strat} | {top_k} | {rerank} | "
                f"**{metrics['hit_rate_at_k'] * 100:.1f}%** | {metrics['recall_at_k'] * 100:.1f}% | "
                f"{metrics['precision_at_k'] * 100:.1f}% | {metrics['mrr']:.4f} | {perf['average_latency_ms']:.1f}ms |"
            )

        return "\n".join(lines)
