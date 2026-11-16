"""API route handlers for query execution and real-time streaming."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from app.api.dependencies import get_query_service
from app.core.exceptions import LLMGenerationError, RetrievalError
from app.schemas.query import QueryRequest, QueryResponse
from app.services.query_service import QueryService

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a knowledge-base query",
    description="Executes a natural language query through the LangGraph workflow with hybrid search, reranking, and citation validation.",
)
def query_knowledge_base(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
):
    try:
        return query_service.execute_query(request)
    except (RetrievalError, LLMGenerationError) as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution error: {str(e)}"
        )


@router.post(
    "/stream",
    summary="Stream query responses via Server-Sent Events",
    description="Streams real-time generation progress and token chunks using Server-Sent Events (SSE).",
)
async def query_knowledge_base_stream(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
):
    try:
        generator = query_service.execute_query_stream(request)
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming query error: {str(e)}"
        )
