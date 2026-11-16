"""Generation quality evaluation metrics: Faithfulness, Answer Relevance, and Citation Correctness."""

import re
from typing import Any, Dict, List, Optional, Set
import numpy as np
from app.retrieval.embeddings import BaseEmbeddingProvider
from app.schemas.citation import Citation
from app.schemas.evaluation import GenerationMetrics


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two vector lists."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def calculate_answer_relevance(
    generated_answer: str,
    expected_answer: str,
    embedding_provider: BaseEmbeddingProvider,
) -> float:
    """Measures semantic alignment between the generated answer and expected ground truth."""
    if not generated_answer or not expected_answer:
        return 0.0

    # Clean citations before vector comparison
    clean_gen = re.sub(r"\[DOC:[^\]]+\]", "", generated_answer).strip()
    clean_exp = re.sub(r"\[DOC:[^\]]+\]", "", expected_answer).strip()

    v_gen = embedding_provider.embed_query(clean_gen)
    v_exp = embedding_provider.embed_query(clean_exp)

    sim = cosine_similarity(v_gen, v_exp)
    return round(float(max(0.0, min(1.0, sim))), 4)


def calculate_faithfulness(
    generated_answer: str,
    retrieved_context: str,
    is_unanswerable: bool = False,
) -> float:
    """Measures groundedness: verifies that claims in the answer exist in the retrieved context."""
    if not generated_answer:
        return 0.0

    # If the ground truth was unanswerable and the model acknowledges insufficient evidence
    if is_unanswerable:
        if "does not contain sufficient information" in generated_answer.lower() or "not contain" in generated_answer.lower():
            return 1.0  # Perfect faithfulness: correctly refused to hallucinate
        return 0.0  # Hallucinated an answer for an unanswerable query

    if not retrieved_context or not retrieved_context.strip():
        return 1.0 if "does not contain" in generated_answer.lower() else 0.0

    # Strip citation tags from generated answer
    clean_answer = re.sub(r"\[DOC:[^\]]+\]", "", generated_answer).strip()
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", clean_answer) if len(s.strip()) > 15]

    if not sentences:
        return 1.0

    context_lower = retrieved_context.lower()
    supported_sentences = 0

    for sent in sentences:
        sent_words = [w.lower() for w in re.findall(r"\b\w{4,}\b", sent)]
        if not sent_words:
            supported_sentences += 1
            continue

        # Check keyword overlap ratio with context
        matches = sum(1 for w in sent_words if w in context_lower)
        overlap_ratio = matches / len(sent_words)

        # Sentence is considered supported if >= 60% of significant content words appear in context
        if overlap_ratio >= 0.60:
            supported_sentences += 1

    return round(float(supported_sentences / len(sentences)), 4)


def calculate_citation_correctness(
    citations: List[Citation],
    relevant_doc_ids: List[str],
) -> Dict[str, float]:
    """Calculates citation validity, precision, and recall against ground truth documents."""
    if not citations and not relevant_doc_ids:
        # Correctly cited nothing for unanswerable question
        return {"citation_validity": 1.0, "citation_precision": 1.0, "citation_recall": 1.0}

    if not citations and relevant_doc_ids:
        return {"citation_validity": 0.0, "citation_precision": 0.0, "citation_recall": 0.0}

    # Citation validity (programmatic check against context blocks)
    valid_citations = sum(1 for c in citations if c.is_valid)
    validity_ratio = valid_citations / len(citations)

    # Citation precision (cited document matches ground-truth document IDs)
    rel_set = set(relevant_doc_ids)
    cited_docs = {c.document_id for c in citations}
    matching_docs = cited_docs.intersection(rel_set)

    precision = len(matching_docs) / len(cited_docs) if cited_docs else 0.0
    recall = len(matching_docs) / len(rel_set) if rel_set else 1.0

    return {
        "citation_validity": round(validity_ratio, 4),
        "citation_precision": round(precision, 4),
        "citation_recall": round(recall, 4),
    }
