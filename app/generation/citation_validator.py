"""Programmatic citation validation and grounding verification."""

import re
from typing import Any, Dict, List, Set, Tuple
from app.core.logging import get_logger
from app.schemas.citation import Citation

logger = get_logger(__name__)


class CitationValidator:
    """Validates that cited sources in LLM responses correspond to actual retrieved context."""

    CITATION_PATTERN = re.compile(r"\[DOC:([^:\]]+):([^\]]+)\]")

    @classmethod
    def extract_citations_from_text(
        cls,
        text: str,
        context_chunks: List[Dict[str, Any]]
    ) -> List[Citation]:
        """Extracts [DOC:doc_id:chunk_id] citations from text and validates against context."""
        chunk_map: Dict[str, Dict[str, Any]] = {
            c.get("chunk_id", ""): c for c in context_chunks
        }
        doc_map: Dict[str, List[Dict[str, Any]]] = {}
        for c in context_chunks:
            doc_id = c.get("document_id", "")
            if doc_id not in doc_map:
                doc_map[doc_id] = []
            doc_map[doc_id].append(c)

        matches = cls.CITATION_PATTERN.findall(text)
        citations: List[Citation] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for doc_id, chunk_id in matches:
            doc_id = doc_id.strip()
            chunk_id = chunk_id.strip()
            pair = (doc_id, chunk_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            if chunk_id in chunk_map:
                matched_chunk = chunk_map[chunk_id]
                chunk_text = matched_chunk.get("text", "")
                excerpt = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                citations.append(
                    Citation(
                        document_id=doc_id,
                        filename=matched_chunk.get("filename", ""),
                        page_number=matched_chunk.get("page_number"),
                        chunk_id=chunk_id,
                        excerpt=excerpt,
                        relevance_score=matched_chunk.get("score"),
                        is_valid=True,
                    )
                )
            else:
                # Chunk ID does not exist in retrieved context -> invalid hallucinated citation
                logger.warning(f"Invalid citation detected: doc_id={doc_id}, chunk_id={chunk_id}")
                citations.append(
                    Citation(
                        document_id=doc_id,
                        filename="unknown",
                        page_number=None,
                        chunk_id=chunk_id,
                        excerpt="[Citation reference not found in retrieved evidence]",
                        relevance_score=0.0,
                        is_valid=False,
                    )
                )

        # Fallback: if model did not use explicit [DOC:..] tags, construct citations from top context chunks
        if not citations and context_chunks:
            for c in context_chunks[:2]:
                chunk_text = c.get("text", "")
                excerpt = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                citations.append(
                    Citation(
                        document_id=c.get("document_id", ""),
                        filename=c.get("filename", ""),
                        page_number=c.get("page_number"),
                        chunk_id=c.get("chunk_id", ""),
                        excerpt=excerpt,
                        relevance_score=c.get("score"),
                        is_valid=True,
                    )
                )

        return citations

    @classmethod
    def validate_citations(
        cls,
        citations: List[Citation],
        allowed_chunk_ids: Set[str]
    ) -> List[Citation]:
        """Flags citations whose chunk_id is not in the allowed retrieved context set."""
        validated: List[Citation] = []
        for cit in citations:
            is_valid = cit.chunk_id in allowed_chunk_ids
            cit.is_valid = is_valid
            validated.append(cit)
        return validated
