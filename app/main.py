"""EnterpriseIQ: Enterprise Knowledge Assistant — FastAPI Application Entry Point."""

import time
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.v1.api_router import api_router
from app.core.config import get_settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentValidationError,
    EnterpriseIQException,
)
from app.core.logging import get_logger

logger = get_logger("enterpriseiq.app")
settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade Enterprise Knowledge Assistant powered by LangGraph, Hybrid Retrieval, and Cross-Encoder Reranking.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TimingMiddleware:
    """Pure ASGI middleware to record request duration header without BaseHTTPMiddleware taskgroups."""

    def __init__(self, app_instance):
        self.app = app_instance

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                headers = list(message.get("headers", []))
                headers.append((b"x-process-time-ms", f"{duration_ms:.2f}".encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


app.add_middleware(TimingMiddleware)


# Global Exception Handlers
@app.exception_handler(EnterpriseIQException)
async def enterpriseiq_exception_handler(request: Request, exc: EnterpriseIQException):
    logger.error(f"Domain error processing {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": exc.__class__.__name__, "message": exc.message, "details": exc.details},
    )


@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "DocumentNotFound", "message": exc.message},
    )


@app.exception_handler(DocumentValidationError)
async def document_validation_handler(request: Request, exc: DocumentValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "DocumentValidationFailed", "message": exc.message},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Request validation failed on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "ValidationError", "details": exc.errors()},
    )


# Include API Routers
app.include_router(api_router)


@app.get("/", tags=["Root"])
def root():
    """Application root endpoint providing system info and links."""
    return {
        "app_name": settings.APP_NAME,
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/api/v1/health",
        "metrics": "/api/v1/metrics",
        "endpoints": {
            "upload_document": "POST /api/v1/documents/upload",
            "list_documents": "GET /api/v1/documents",
            "query": "POST /api/v1/query",
            "stream_query": "POST /api/v1/query/stream",
            "evaluation_results": "GET /api/v1/evaluate/results",
        },
    }
