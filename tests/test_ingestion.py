"""Unit tests for document loaders, text cleaning, and parsing."""

import pytest
from app.core.exceptions import DocumentValidationError
from app.ingestion.loaders import DocumentLoaderFactory, TextLoader
from app.ingestion.parsers import TextCleaner
from app.schemas.document import DocumentFileType


def test_text_cleaner_whitespace_and_artifacts():
    """Verify whitespace collapsing and non-printable character removal."""
    dirty_text = "Line 1\r\n\r\n\r\n\r\nLine 2   with   spaces\xa0and\x00null"
    cleaned = TextCleaner.clean_text(dirty_text)

    assert "\r" not in cleaned
    assert "\x00" not in cleaned
    assert "\xa0" not in cleaned
    assert "Line 1\n\nLine 2 with spaces andnull" == cleaned


def test_text_cleaner_extract_heading():
    """Verify heading extraction from Markdown and title text."""
    md_heading = TextCleaner.extract_heading("# System Architecture\nDetails follow.")
    assert md_heading == "System Architecture"

    title_heading = TextCleaner.extract_heading("COMPANY EXPENSE POLICY\nAll receipts must be submitted.")
    assert title_heading == "COMPANY EXPENSE POLICY"

    empty_heading = TextCleaner.extract_heading("")
    assert empty_heading is None


def test_loader_factory_extension_matching():
    """Verify loader factory selects correct loader and enum type."""
    loader, file_type = DocumentLoaderFactory.get_loader("sample.pdf")
    assert file_type == DocumentFileType.PDF

    loader, file_type = DocumentLoaderFactory.get_loader("doc.docx")
    assert file_type == DocumentFileType.DOCX

    loader, file_type = DocumentLoaderFactory.get_loader("notes.txt")
    assert file_type == DocumentFileType.TXT
    assert isinstance(loader, TextLoader)

    loader, file_type = DocumentLoaderFactory.get_loader("README.md")
    assert file_type == DocumentFileType.MD

    with pytest.raises(DocumentValidationError):
        DocumentLoaderFactory.get_loader("unsupported.xyz")


def test_content_hashing():
    """Verify SHA-256 calculation produces deterministic 64-char hex strings."""
    data1 = b"Hello world"
    data2 = b"Hello world"
    data3 = b"Different content"

    hash1 = DocumentLoaderFactory.calculate_sha256(data1)
    hash2 = DocumentLoaderFactory.calculate_sha256(data2)
    hash3 = DocumentLoaderFactory.calculate_sha256(data3)

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64
