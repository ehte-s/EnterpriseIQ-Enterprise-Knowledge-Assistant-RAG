"""Answer generation service supporting configurable LLM backends and mock fallbacks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from app.core.config import get_settings
from app.core.exceptions import LLMGenerationError
from app.core.logging import get_logger
from app.generation.prompts import REFORMULATION_PROMPT, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from app.schemas.query import TokenUsage

logger = get_logger(__name__)


class BaseAnswerGenerator(ABC):
    """Abstract interface for LLM answer generation and query reformulation."""

    @abstractmethod
    def generate_answer(
        self,
        query: str,
        context_blocks: str,
    ) -> Tuple[str, TokenUsage]:
        """Generates grounded answer from context."""
        pass

    @abstractmethod
    def reformulate_query(
        self,
        query: str,
        previous_query: str,
    ) -> str:
        """Reformulates a search query to broaden lexical and semantic coverage."""
        pass


class LLMAnswerGenerator(BaseAnswerGenerator):
    """OpenAI-compatible LLM answer generator."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.MODEL_NAME
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            if not self.api_key:
                raise LLMGenerationError("API key not configured for LLM answer generator.")
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate_answer(
        self,
        query: str,
        context_blocks: str,
    ) -> Tuple[str, TokenUsage]:
        client = self._get_client()
        user_content = USER_PROMPT_TEMPLATE.format(context_blocks=context_blocks, query=query)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.temperature,
            )
            answer = response.choices[0].message.content or ""
            usage = response.usage
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                estimated_cost_usd=round((usage.total_tokens if usage else 0) * 0.000002, 6),
            )
            return answer, token_usage
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMGenerationError(f"LLM generation error: {e}") from e

    def reformulate_query(
        self,
        query: str,
        previous_query: str,
    ) -> str:
        client = self._get_client()
        prompt = REFORMULATION_PROMPT.format(query=query, current_retrieval_query=previous_query)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a search query optimizer. Return only the rewritten search query."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=60,
            )
            rewritten = response.choices[0].message.content or query
            return rewritten.strip().strip('"')
        except Exception as e:
            logger.warning(f"LLM query reformulation failed ({e}), using keyword heuristic.")
            return f"{query} policy requirements standards"


class MockAnswerGenerator(BaseAnswerGenerator):
    """Deterministic answer generator for testing without live LLM calls."""

    def generate_answer(
        self,
        query: str,
        context_blocks: str,
    ) -> Tuple[str, TokenUsage]:
        if not context_blocks or not context_blocks.strip():
            return (
                "The provided enterprise documentation does not contain sufficient information to answer this question.",
                TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70, estimated_cost_usd=0.00014)
            )

        # Extract first tagged block e.g. [DOC:doc_id:chunk_id]
        import re
        tag_match = re.search(r"(\[DOC:[^:]+:[^\]]+\])", context_blocks)
        tag = tag_match.group(1) if tag_match else "[DOC:doc_sample:c0001]"

        # Extract first sentence of the context text
        clean_text = re.sub(r"\[DOC:[^\]]+\]", "", context_blocks).strip()
        first_sentence = clean_text.split(".")[0] + "." if "." in clean_text else clean_text[:150]

        answer = f"Based on the enterprise documentation: {first_sentence} {tag}"
        token_usage = TokenUsage(
            prompt_tokens=150,
            completion_tokens=40,
            total_tokens=190,
            estimated_cost_usd=0.00038,
        )
        return answer, token_usage

    def reformulate_query(
        self,
        query: str,
        previous_query: str,
    ) -> str:
        # Append policy domain synonyms
        return f"{query} policy rules guidelines"


class AnswerGeneratorFactory:
    """Factory to instantiate answer generator."""

    @staticmethod
    def get_generator(use_mock: bool = False) -> BaseAnswerGenerator:
        settings = get_settings()
        if use_mock or not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "mock-key-for-local-testing":
            return MockAnswerGenerator()
        return LLMAnswerGenerator()
