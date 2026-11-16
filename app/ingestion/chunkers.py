"""Configurable document chunking strategies."""

import re
from abc import ABC, abstractmethod
from typing import List, Optional
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split input text into a list of chunk strings."""
        pass


class RecursiveCharacterChunker(BaseChunker):
    """Recursively splits text on hierarchical separators: paragraphs, lines, sentences, spaces."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        return self._split(text.strip(), self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []

        # Find the first separator present in the text
        separator = separators[-1]
        new_separators: List[str] = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits: List[str] = []
        for s in splits:
            if separator != "" and separator != " ":
                # Retain separator context if useful
                piece = s
            else:
                piece = s

            if len(piece) <= self.chunk_size:
                good_splits.append(piece)
            else:
                if new_separators:
                    sub_splits = self._split(piece, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    # Fallback slice if no further separators exist
                    for j in range(0, len(piece), self.chunk_size - self.chunk_overlap):
                        good_splits.append(piece[j:j + self.chunk_size])

        # Merge small splits back up to chunk_size respecting chunk_overlap
        current_chunk: List[str] = []
        current_len = 0

        for piece in good_splits:
            piece_len = len(piece)
            sep_len = len(separator) if current_chunk else 0

            if current_len + sep_len + piece_len > self.chunk_size and current_chunk:
                chunk_str = separator.join(current_chunk).strip()
                if chunk_str:
                    final_chunks.append(chunk_str)

                # Keep overlap pieces from the end of current_chunk
                overlap_accum: List[str] = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= self.chunk_overlap:
                        overlap_accum.insert(0, item)
                        overlap_len += len(item) + len(separator)
                    else:
                        break
                current_chunk = overlap_accum
                current_len = sum(len(x) for x in current_chunk) + (len(separator) * (len(current_chunk) - 1) if current_chunk else 0)

            current_chunk.append(piece)
            current_len += piece_len + (len(separator) if len(current_chunk) > 1 else 0)

        if current_chunk:
            chunk_str = separator.join(current_chunk).strip()
            if chunk_str:
                final_chunks.append(chunk_str)

        return final_chunks


class FixedSizeChunker(BaseChunker):
    """Splits text into fixed character windows with sliding overlap."""

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        text = text.strip()
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if i + self.chunk_size >= len(text):
                break
        return chunks


class StructureAwareChunker(BaseChunker):
    """Splits text on Markdown and document section headers, then falls back to recursive."""

    HEADER_PATTERN = re.compile(r"(^(?:#{1,6}\s+.*|[A-Z0-9\.\s]{3,60}:?\n=+\n?))", re.MULTILINE)

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self.fallback = RecursiveCharacterChunker(chunk_size, chunk_overlap)

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        # Split on header pattern
        sections = self.HEADER_PATTERN.split(text)
        chunks: List[str] = []

        current_heading = ""
        for part in sections:
            part = part.strip()
            if not part:
                continue

            if self.HEADER_PATTERN.match(part):
                current_heading = part
            else:
                full_text = f"{current_heading}\n\n{part}".strip() if current_heading else part
                if len(full_text) <= self.chunk_size:
                    chunks.append(full_text)
                else:
                    chunks.extend(self.fallback.split_text(full_text))

        return chunks if chunks else self.fallback.split_text(text)


class ChunkerFactory:
    """Factory helper to obtain chunkers for experiments."""

    @staticmethod
    def create(
        strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> BaseChunker:
        strat = strategy.lower().strip()
        if strat in ("recursive", "recursive_character"):
            return RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif strat in ("fixed", "fixed_size"):
            return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif strat in ("structure", "structure_aware"):
            return StructureAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            raise ValueError(f"Unknown chunking strategy: '{strategy}'")
