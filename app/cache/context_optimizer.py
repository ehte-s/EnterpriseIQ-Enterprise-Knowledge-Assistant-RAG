"""Context optimization engine for duplicate pruning, relevance filtering, and token budgeting."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from app.core.logging import get_logger

logger = get_logger(__name__)


class ContextOptimizer:
    """Optimizes context chunks before prompt construction to minimize token expenditure."""

    def __init__(
        self,
        max_context_tokens: int = 1500,
        similarity_threshold: float = 0.80,
        min_relevance_score: float = 0.25,
    ):
        self.max_context_tokens = max_context_tokens
        self.similarity_threshold = similarity_threshold
        self.min_relevance_score = min_relevance_score

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic token estimation: ~1 token per 4 characters / 0.75 words."""
        return max(1, int(len(text) / 3.8))

    @staticmethod
    def _jaccard_similarity(text1: str, text2: str) -> float:
        """Calculates token-level Jaccard similarity between two passages."""
        words1 = set(re.findall(r"\b\w{3,}\b", text1.lower()))
        words2 = set(re.findall(r"\b\w{3,}\b", text2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)

    def optimize_context(
        self,
        chunks: List[Dict[str, Any]],
        token_budget: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Filters, deduplicates, and trims context chunks to fit within a strict token budget.

        Returns:
            Tuple of (optimized_chunks, optimization_telemetry)
        """
        budget = token_budget or self.max_context_tokens
        if not chunks:
            return [], {"raw_tokens": 0, "optimized_tokens": 0, "tokens_saved": 0, "savings_pct": 0.0}

        # 1. Filter out chunks below minimum relevance threshold
        filtered = [
            c for c in chunks
            if c.get("score", 0.0) >= self.min_relevance_score
            or c.get("reranker_score", 0.0) >= self.min_relevance_score
        ]
        if not filtered and chunks:
            # Keep top chunk if everything fell below threshold to avoid complete starvation
            filtered = [chunks[0]]

        # 2. Sort by descending relevance
        filtered.sort(
            key=lambda c: c.get("reranker_score", c.get("score", 0.0)),
            reverse=True
        )

        # 3. Deduplication and near-duplicate removal
        deduplicated: List[Dict[str, Any]] = []
        raw_tokens_total = sum(self.estimate_tokens(c.get("text", "")) for c in chunks)

        for candidate in filtered:
            candidate_text = candidate.get("text", "").strip()
            if not candidate_text:
                continue

            # Check overlap with already accepted chunks
            is_redundant = False
            for accepted in deduplicated:
                sim = self._jaccard_similarity(candidate_text, accepted.get("text", ""))
                if sim >= self.similarity_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                deduplicated.append(candidate)

        # 4. Strict Token Budget Enforcement
        optimized_chunks: List[Dict[str, Any]] = []
        accumulated_tokens = 0

        for chunk in deduplicated:
            chunk_tokens = self.estimate_tokens(chunk.get("text", ""))
            if accumulated_tokens + chunk_tokens <= budget:
                optimized_chunks.append(chunk)
                accumulated_tokens += chunk_tokens
            else:
                # If budget allows at least 60 tokens, include a truncated excerpt
                remaining_budget = budget - accumulated_tokens
                if remaining_budget >= 60 and not optimized_chunks:
                    # Truncate text to remaining budget
                    char_limit = int(remaining_budget * 3.8)
                    truncated_chunk = dict(chunk)
                    truncated_chunk["text"] = chunk["text"][:char_limit] + "..."
                    optimized_chunks.append(truncated_chunk)
                    accumulated_tokens += remaining_budget
                break

        tokens_saved = max(0, raw_tokens_total - accumulated_tokens)
        savings_pct = round((tokens_saved / raw_tokens_total * 100.0), 2) if raw_tokens_total > 0 else 0.0

        telemetry = {
            "raw_chunks_count": len(chunks),
            "optimized_chunks_count": len(optimized_chunks),
            "raw_tokens": raw_tokens_total,
            "optimized_tokens": accumulated_tokens,
            "tokens_saved": tokens_saved,
            "savings_pct": savings_pct,
        }

        logger.info(
            f"[ContextOptimizer] Trimmed {len(chunks)} -> {len(optimized_chunks)} chunks. "
            f"Tokens: {raw_tokens_total} -> {accumulated_tokens} (Saved {tokens_saved} tokens / {savings_pct}%)"
        )
        return optimized_chunks, telemetry
