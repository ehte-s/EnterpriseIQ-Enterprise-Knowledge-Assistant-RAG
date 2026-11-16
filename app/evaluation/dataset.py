"""Dataset loader and validator for the 100-question RAG evaluation benchmark."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from app.core.exceptions import EnterpriseIQException
from app.core.logging import get_logger
from app.schemas.evaluation import EvaluationQuestion

logger = get_logger(__name__)


class EvaluationDataset:
    """Manages loading, filtering, and validation of evaluation benchmark datasets."""

    def __init__(self, dataset_path: Optional[str] = None):
        self.path = Path(dataset_path or "datasets/evaluation_questions.jsonl")
        self.questions: List[EvaluationQuestion] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            raise EnterpriseIQException(f"Evaluation dataset not found at: {self.path}")

        loaded: List[EvaluationQuestion] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    loaded.append(EvaluationQuestion(**data))
                except Exception as e:
                    logger.error(f"Error parsing line {line_num} in {self.path}: {e}")
                    raise EnterpriseIQException(f"Invalid dataset format at line {line_num}: {e}") from e

        self.questions = loaded
        logger.info(f"Loaded {len(self.questions)} evaluation questions from {self.path}")

    def filter_by_category(self, category: str) -> List[EvaluationQuestion]:
        """Returns questions belonging to a specific category."""
        return [q for q in self.questions if q.category.lower() == category.lower()]

    def get_answerable_questions(self) -> List[EvaluationQuestion]:
        """Returns only questions that have at least one relevant ground-truth document."""
        return [q for q in self.questions if len(q.relevant_document_ids) > 0]

    def get_unanswerable_questions(self) -> List[EvaluationQuestion]:
        """Returns queries that have no answer in the knowledge base (fallback test)."""
        return [q for q in self.questions if len(q.relevant_document_ids) == 0]

    def get_category_counts(self) -> Dict[str, int]:
        """Returns count of questions per category."""
        counts: Dict[str, int] = {}
        for q in self.questions:
            counts[q.category] = counts.get(q.category, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)
