"""Text cleaning, normalization, and document metadata extraction utilities."""

import re
import unicodedata
from typing import Optional


class TextCleaner:
    """Provides text normalization and cleaning for raw document extracts."""

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapses excessive horizontal whitespace while preserving paragraph line breaks."""
        # Replace non-breaking spaces and exotic spaces
        text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple blank lines to at most two newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple horizontal spaces to single space
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def clean_text(text: str) -> str:
        """Full cleaning pipeline for raw extracted document text."""
        if not text:
            return ""

        # Normalize unicode (NFKC)
        text = unicodedata.normalize("NFKC", text)

        # Remove null bytes and non-printable control characters (except newline, tab)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Normalize whitespace
        text = TextCleaner.normalize_whitespace(text)

        return text

    @staticmethod
    def extract_heading(text: str) -> Optional[str]:
        """Attempts to detect section heading from Markdown or capitalized text."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return None

        first_line = lines[0]
        # Markdown heading check
        if first_line.startswith("#"):
            return first_line.lstrip("#").strip()

        # Short uppercase heading check
        if len(first_line) < 80 and (first_line.isupper() or first_line.istitle()):
            return first_line

        return None
