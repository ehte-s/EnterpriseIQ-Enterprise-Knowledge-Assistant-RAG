"""Health check and telemetry metrics endpoints."""

import os
import psutil
from fastapi import APIRouter, Depends, status
from app.api.dependencies import get_document_service, get_query_service, get_vector_store
from app.core.config import get_settings
from app.retrieval.vector_store import QdrantVectorStore
from app.services.document_service import DocumentService
from app.services.query_service import QueryService

router = APIRouter(tags=["Health & Metrics"])


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Application Health Check",
    description="Returns the operational status of EnterpriseIQ components.",
)
def health_check(
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    document_service: DocumentService = Depends(get_document_service),
):
    settings = get_settings()
    vector_store_healthy = False
    try:
        points_count = vector_store.count_chunks()
        vector_store_healthy = True
    except Exception:
        points_count = 0

    return {
        "status": "healthy" if vector_store_healthy else "degraded",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "components": {
            "vector_store": "online" if vector_store_healthy else "offline",
            "points_indexed": points_count,
            "documents_indexed": len(document_service.list_documents()),
        },
    }


@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Application Telemetry Metrics",
    description="Exposes query throughput, latency, token consumption, and cache metrics.",
)
def get_metrics(
    query_service: QueryService = Depends(get_query_service),
    document_service: DocumentService = Depends(get_document_service),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    query_metrics = query_service.get_metrics()
    return {
        "system": {
            "cpu_percent": process.cpu_percent(interval=None),
            "memory_usage_mb": round(memory_info.rss / (1024 * 1024), 2),
        },
        "knowledge_base": {
            "total_documents": len(document_service.list_documents()),
            "total_chunks": vector_store.count_chunks(),
        },
        "query_performance": query_metrics,
    }
