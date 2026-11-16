"""API Client for EnterpriseIQ Streamlit Frontend.

Encapsulates all HTTP communication with the FastAPI backend, ensuring
a clean separation of concerns with zero business logic in the UI layer.
"""

import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional
import httpx
import requests

logger = logging.getLogger(__name__)

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")


class EnterpriseIQClient:
    """Client for interacting with the EnterpriseIQ REST API."""

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Check system health and obtain uptime/metadata."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        """Retrieve Qdrant, BM25, and semantic cache metrics."""
        try:
            resp = requests.get(f"{self.base_url}/health/metrics", timeout=5.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch metrics: {e}")
            return {}

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed enterprise documents."""
        try:
            resp = requests.get(f"{self.base_url}/documents", timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("documents", [])
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    def upload_document(
        self,
        file_bytes: bytes,
        filename: str,
        chunker_strategy: str = "structure_aware",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> Dict[str, Any]:
        """Upload and index a document via multipart form data."""
        url = f"{self.base_url}/documents/upload"
        files = {"file": (filename, file_bytes)}
        data = {
            "chunker_strategy": chunker_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        try:
            resp = requests.post(url, files=files, data=data, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            error_detail = "Upload failed"
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = str(e)
            return {"error": error_detail, "status_code": e.response.status_code if e.response else 500}
        except Exception as e:
            return {"error": str(e), "status_code": 500}

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document, its vector chunks, and evict dependent cache entries."""
        try:
            resp = requests.delete(f"{self.base_url}/documents/{document_id}", timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def query(
        self,
        query_text: str,
        retrieval_strategy: str = "hybrid",
        top_k: int = 5,
        use_cache: bool = True,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit a query for RAG execution."""
        url = f"{self.base_url}/query"
        payload = {
            "query": query_text,
            "retrieval_strategy": retrieval_strategy,
            "top_k": top_k,
            "use_cache": use_cache,
            "conversation_id": conversation_id,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            error_detail = "Query failed"
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = str(e)
            return {"error": error_detail, "answer": f"Error: {error_detail}", "citations": []}
        except Exception as e:
            return {"error": str(e), "answer": f"Connection error: {e}", "citations": []}

    def query_stream(
        self,
        query_text: str,
        retrieval_strategy: str = "hybrid",
        top_k: int = 5,
        use_cache: bool = True,
        conversation_id: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream query responses via Server-Sent Events (SSE)."""
        url = f"{self.base_url}/query/stream"
        payload = {
            "query": query_text,
            "retrieval_strategy": retrieval_strategy,
            "top_k": top_k,
            "use_cache": use_cache,
            "conversation_id": conversation_id,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", url, json=payload) as response:
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str:
                                try:
                                    event = json.loads(data_str)
                                    yield event
                                except json.JSONDecodeError:
                                    yield {"type": "token", "content": data_str}
        except Exception as e:
            yield {"type": "error", "error": str(e)}

    def get_evaluation_results(self) -> Dict[str, Any]:
        """Fetch latest reproducible evaluation benchmark metrics."""
        try:
            resp = requests.get(f"{self.base_url}/evaluate/results", timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Could not load evaluation results: {e}")
            return {}
