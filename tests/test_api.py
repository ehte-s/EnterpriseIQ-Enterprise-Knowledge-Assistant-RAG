"""Integration tests for EnterpriseIQ FastAPI REST endpoints using TestClient."""

import io
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient for FastAPI endpoints."""
    return TestClient(app)


def test_root_endpoint(client):
    """Verify application root metadata endpoint."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["app_name"] == "EnterpriseIQ"
    assert data["documentation"] == "/docs"
    assert "x-process-time-ms" in resp.headers


def test_health_check(client):
    """Verify health endpoint reports online components."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded")
    assert "components" in data


def test_metrics_endpoint(client):
    """Verify telemetry metrics endpoint."""
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "system" in data
    assert "knowledge_base" in data
    assert "query_performance" in data


def test_document_lifecycle_and_query(client, sample_text_document):
    """Verify upload, list, query, and delete workflow."""
    # 1. Upload document
    file_bytes = io.BytesIO(sample_text_document.encode("utf-8"))
    upload_resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test_vacation_policy.txt", file_bytes, "text/plain")},
    )
    assert upload_resp.status_code == 201
    upload_data = upload_resp.json()
    doc_id = upload_data["document_id"]
    assert doc_id.startswith("doc_")
    assert upload_data["chunk_count"] > 0

    # 2. List documents
    list_resp = client.get("/api/v1/documents")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total_documents"] >= 1
    assert any(d["document_id"] == doc_id for d in list_data["documents"])

    # 3. Get document by ID
    get_resp = client.get(f"/api/v1/documents/{doc_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["document_id"] == doc_id

    # 4. Query knowledge base
    query_payload = {
        "query": "How many days of annual vacation leave do employees receive?",
        "retrieval_strategy": "hybrid",
        "top_k": 3,
        "use_reranker": False,
        "use_cache": False,
    }
    query_resp = client.post("/api/v1/query", json=query_payload)
    assert query_resp.status_code == 200
    query_data = query_resp.json()
    assert query_data["query"] == query_payload["query"]
    assert query_data["status"] == "success"
    assert query_data["confidence"] > 0.0
    assert "metadata" in query_data

    # 5. Query streaming endpoint
    stream_resp = client.post("/api/v1/query/stream", json=query_payload)
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    assert "event: status" in stream_resp.text
    assert "event: complete" in stream_resp.text

    # 6. Delete document
    del_resp = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 200

    # 7. Confirm deletion
    get_after_del = client.get(f"/api/v1/documents/{doc_id}")
    assert get_after_del.status_code == 404


def test_evaluation_results_endpoint(client):
    """Verify evaluation results retrieval endpoint."""
    resp = client.get("/api/v1/evaluate/results")
    assert resp.status_code == 200
    data = resp.json()
    assert "baseline" in data
    assert "optimized" in data
