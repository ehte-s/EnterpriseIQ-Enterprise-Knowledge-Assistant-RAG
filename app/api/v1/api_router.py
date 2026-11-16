"""Aggregates all API v1 routes."""

from fastapi import APIRouter
from app.api.v1.routes import documents, evaluation, health, query

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(query.router)
api_router.include_router(evaluation.router)
