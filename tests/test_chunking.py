"""Unit tests for document chunking strategies."""

import pytest
from app.ingestion.chunkers import (
    BaseChunker,
    ChunkerFactory,
    FixedSizeChunker,
    RecursiveCharacterChunker,
    StructureAwareChunker,
)


def test_chunker_invalid_overlap():
    """Verify that overlap >= chunk_size raises ValueError."""
    with pytest.raises(ValueError):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=50, chunk_overlap=60)


def test_recursive_character_chunker_basic(sample_text_document):
    """Test recursive character chunker produces non-empty chunks within size limits."""
    chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.split_text(sample_text_document)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 250  # Allows slight boundary buffer if word unbroken
        assert len(chunk.strip()) > 0


def test_fixed_size_chunker():
    """Test fixed size chunker creates sequential sliding window chunks."""
    text = "A" * 500
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_text(text)

    assert len(chunks) == 6  # 500 length with step 80 -> ceil(500/80)
    assert all(len(c) <= 100 for c in chunks)


def test_structure_aware_chunker(sample_markdown_document):
    """Test structure-aware chunker preserves section headers."""
    chunker = StructureAwareChunker(chunk_size=250, chunk_overlap=30)
    chunks = chunker.split_text(sample_markdown_document)

    assert len(chunks) >= 3
    # Check that headers appear in chunks
    has_header = any("Password Requirements" in c or "Multi-Factor" in c for c in chunks)
    assert has_header


def test_chunker_factory():
    """Verify ChunkerFactory instantiates proper classes."""
    c1 = ChunkerFactory.create("recursive", chunk_size=300, chunk_overlap=40)
    assert isinstance(c1, RecursiveCharacterChunker)
    assert c1.chunk_size == 300
    assert c1.chunk_overlap == 40

    c2 = ChunkerFactory.create("fixed", chunk_size=400, chunk_overlap=50)
    assert isinstance(c2, FixedSizeChunker)

    c3 = ChunkerFactory.create("structure", chunk_size=500, chunk_overlap=50)
    assert isinstance(c3, StructureAwareChunker)

    with pytest.raises(ValueError):
        ChunkerFactory.create("unknown_strategy")
