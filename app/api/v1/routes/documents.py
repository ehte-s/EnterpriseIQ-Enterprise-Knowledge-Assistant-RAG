"""API route handlers for document upload, indexing, listing, and deletion."""

from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from app.api.dependencies import get_document_service, get_hybrid_retriever
from app.core.exceptions import DocumentNotFoundError, DocumentValidationError
from app.schemas.document import (
    DocumentListResponse,
    DocumentMetadata,
    DocumentUploadResponse,
)
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
    description="Uploads a PDF, DOCX, TXT, or Markdown document, cleans text, generates embeddings, and indexes into Qdrant.",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (PDF, DOCX, TXT, or MD)"),
    document_service: DocumentService = Depends(get_document_service),
    retriever=Depends(get_hybrid_retriever),
):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{file.filename}' is empty."
            )

        response = document_service.index_file(
            file_bytes=content,
            filename=file.filename or "uploaded_document.txt",
        )
        # Sync BM25 index with updated vector store
        retriever.sync_bm25_index()
        return response
    except DocumentValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


class IndexDocumentRequest(BaseModel):
    file_path: str
    document_id: Optional[str] = None


@router.post(
    "/index",
    response_model=DocumentUploadResponse,
    summary="Index document from file path",
    description="Indexes a document from an existing filesystem path.",
)
def index_document_from_path(
    req: IndexDocumentRequest,
    document_service: DocumentService = Depends(get_document_service),
    retriever=Depends(get_hybrid_retriever),
):
    from pathlib import Path
    p = Path(req.file_path)
    if not p.exists() or not p.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File at '{req.file_path}' does not exist."
        )

    try:
        with open(p, "rb") as f:
            content = f.read()

        response = document_service.index_file(
            file_bytes=content,
            filename=p.name,
            document_id=req.document_id,
        )
        retriever.sync_bm25_index()
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
    description="Retrieves metadata for all documents currently indexed in the knowledge base.",
)
def list_documents(
    document_service: DocumentService = Depends(get_document_service),
):
    docs = document_service.list_documents()
    return DocumentListResponse(
        total_documents=len(docs),
        documents=docs,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentMetadata,
    summary="Get document metadata",
    description="Retrieves metadata for a specific document by its unique ID.",
)
def get_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    try:
        return document_service.get_document(document_id)
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a document and its chunks",
    description="Deletes a document from the registry and purges all its chunks from the Qdrant vector database and BM25 index.",
)
def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
    retriever=Depends(get_hybrid_retriever),
):
    try:
        document_service.delete_document(document_id)
        retriever.sync_bm25_index()
        return {"status": "success", "message": f"Document '{document_id}' successfully deleted."}
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
