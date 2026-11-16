"""BM25 sparse keyword retrieval engine for EnterpriseIQ."""

import re
import string
from typing import Any, Dict, List, Optional
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Standard English stop words
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


class BM25Searcher:
    """Okapi BM25 keyword search index."""

    def __init__(self):
        self._bm25 = None
        self._corpus_chunks: List[Dict[str, Any]] = []
        self._tokenized_corpus: List[List[str]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Custom tokenizer: lowercase, strips punctuation, removes stopwords."""
        # Lowercase and clean
        text = text.lower()
        # Remove punctuation
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Builds or rebuilds BM25 index from a list of chunk payload dictionaries."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise RetrievalError("rank_bm25 package is required for BM25 keyword search.")

        self._corpus_chunks = chunks
        self._tokenized_corpus = [self.tokenize(c.get("text", "")) for c in chunks]

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            logger.info(f"Built BM25 index with {len(self._corpus_chunks)} documents.")
        else:
            self._bm25 = None

    def add_chunks(self, new_chunks: List[Dict[str, Any]]):
        """Appends new chunks to index and updates BM25 corpus."""
        all_chunks = self._corpus_chunks + new_chunks
        self.build_index(all_chunks)

    def delete_document(self, document_id: str):
        """Removes all chunks belonging to document_id and rebuilds index."""
        filtered_chunks = [c for c in self._corpus_chunks if c.get("document_id") != document_id]
        self.build_index(filtered_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Executes BM25 score calculation and returns ranked chunks."""
        if not self._bm25 or not self._corpus_chunks:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # Pair scores with chunk payloads
        results = []
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue

            chunk = self._corpus_chunks[idx]

            # Apply filters if provided
            if filters:
                match = True
                for k, v in filters.items():
                    if chunk.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            # Normalize BM25 score to 0..1 scale
            normalized_score = float(score / max_score)
            results.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "document_id": chunk.get("document_id", ""),
                "filename": chunk.get("filename", ""),
                "page_number": chunk.get("page_number"),
                "section_heading": chunk.get("section_heading"),
                "chunk_index": chunk.get("chunk_index", 0),
                "text": chunk.get("text", ""),
                "score": normalized_score,
                "raw_bm25_score": float(score),
            })

        # Sort descending by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
