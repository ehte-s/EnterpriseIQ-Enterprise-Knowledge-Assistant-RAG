"""Shared test fixtures for EnterpriseIQ test suite."""

import io
import pytest
from app.ingestion.chunkers import RecursiveCharacterChunker
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.vector_store import QdrantVectorStore


@pytest.fixture
def mock_embedding():
    """Provides a deterministic 384-dimensional MockEmbedding."""
    return MockEmbedding(dimension=384)


@pytest.fixture
def in_memory_vector_store(mock_embedding):
    """Provides an isolated in-memory QdrantVectorStore."""
    return QdrantVectorStore(
        collection_name="test_enterpriseiq",
        dimension=mock_embedding.dimension,
        in_memory=True
    )


@pytest.fixture
def sample_text_document() -> str:
    """Sample enterprise policy text for testing."""
    return """ACME CORPORATION LEAVE POLICY

1. Annual Leave
Employees are entitled to 25 days of paid annual leave per calendar year. Leave must be requested at least two weeks in advance through the HR portal.

2. Sick Leave
Employees receive 10 days of paid sick leave annually. A medical certificate is required for absences exceeding three consecutive days.

3. Parental Leave
Primary caregivers are eligible for 16 weeks of fully paid parental leave. Secondary caregivers receive 6 weeks of fully paid leave.
"""


@pytest.fixture
def sample_markdown_document() -> str:
    """Sample markdown document with structural headings."""
    return """# IT Security Policy

## 1. Password Requirements
All passwords must be at least 14 characters long and include numbers and special characters. Passwords must be rotated every 90 days.

## 2. Multi-Factor Authentication
MFA is mandatory for all internal and external access to company systems. SMS verification is prohibited; employees must use hardware keys or authenticator apps.

## 3. Remote Work & VPN
All corporate laptops must connect to the corporate VPN when accessing internal services from public networks.
"""
