"""Document loaders for PDF, DOCX, TXT, and Markdown files."""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from app.core.exceptions import DocumentParsingError, DocumentValidationError
from app.core.logging import get_logger
from app.schemas.document import DocumentFileType

logger = get_logger(__name__)


@dataclass
class LoadedPage:
    """Represents text extracted from a document segment or page."""
    text: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    metadata: Optional[Dict[str, Union[str, int, float, bool]]] = None


class BaseDocumentLoader(ABC):
    """Abstract base document loader."""

    @abstractmethod
    def load(self, file_path_or_bytes: Union[str, Path, bytes], filename: str) -> List[LoadedPage]:
        """Loads and extracts text pages/segments from document."""
        pass


class PDFLoader(BaseDocumentLoader):
    """PDF document loader with page-level extraction."""

    def load(self, file_path_or_bytes: Union[str, Path, bytes], filename: str) -> List[LoadedPage]:
        try:
            import pypdf
        except ImportError:
            raise DocumentParsingError("pypdf is required to parse PDF documents.")

        import io
        pages: List[LoadedPage] = []

        try:
            if isinstance(file_path_or_bytes, (str, Path)):
                stream = open(file_path_or_bytes, "rb")
            else:
                stream = io.BytesIO(file_path_or_bytes)

            with stream:
                reader = pypdf.PdfReader(stream)
                total_pages = len(reader.pages)
                for page_idx in range(total_pages):
                    page = reader.pages[page_idx]
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(
                            LoadedPage(
                                text=text,
                                page_number=page_idx + 1,
                                metadata={"total_pages": total_pages}
                            )
                        )
            if not pages:
                logger.warning(f"No extractable text found in PDF: {filename}")
            return pages
        except Exception as e:
            logger.error(f"Failed to parse PDF {filename}: {str(e)}")
            raise DocumentParsingError(f"Error parsing PDF '{filename}': {str(e)}") from e


class DocxLoader(BaseDocumentLoader):
    """DOCX loader preserving paragraph and heading structure."""

    def load(self, file_path_or_bytes: Union[str, Path, bytes], filename: str) -> List[LoadedPage]:
        try:
            import docx
        except ImportError:
            raise DocumentParsingError("python-docx is required to parse DOCX documents.")

        import io
        try:
            if isinstance(file_path_or_bytes, (str, Path)):
                doc = docx.Document(file_path_or_bytes)
            else:
                doc = docx.Document(io.BytesIO(file_path_or_bytes))

            pages: List[LoadedPage] = []
            current_heading: Optional[str] = None
            current_paragraphs: List[str] = []

            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue

                if paragraph.style and paragraph.style.name and paragraph.style.name.startswith("Heading"):
                    if current_paragraphs:
                        pages.append(
                            LoadedPage(
                                text="\n\n".join(current_paragraphs),
                                section_heading=current_heading,
                                page_number=1
                            )
                        )
                        current_paragraphs = []
                    current_heading = text
                else:
                    current_paragraphs.append(text)

            # Process tables in docx
            for table in doc.tables:
                table_rows: List[str] = []
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells]
                    table_rows.append(" | ".join(row_text))
                if table_rows:
                    current_paragraphs.append("\n".join(table_rows))

            if current_paragraphs:
                pages.append(
                    LoadedPage(
                        text="\n\n".join(current_paragraphs),
                        section_heading=current_heading,
                        page_number=1
                    )
                )

            return pages
        except Exception as e:
            logger.error(f"Failed to parse DOCX {filename}: {str(e)}")
            raise DocumentParsingError(f"Error parsing DOCX '{filename}': {str(e)}") from e


class TextLoader(BaseDocumentLoader):
    """Loader for Plain Text and Markdown documents."""

    def load(self, file_path_or_bytes: Union[str, Path, bytes], filename: str) -> List[LoadedPage]:
        try:
            if isinstance(file_path_or_bytes, (str, Path)):
                with open(file_path_or_bytes, "rb") as f:
                    content_bytes = f.read()
            else:
                content_bytes = file_path_or_bytes

            # Attempt UTF-8, then fallback to latin-1
            try:
                text = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = content_bytes.decode("latin-1")

            return [LoadedPage(text=text, page_number=1)]
        except Exception as e:
            logger.error(f"Failed to read text file {filename}: {str(e)}")
            raise DocumentParsingError(f"Error reading file '{filename}': {str(e)}") from e


class DocumentLoaderFactory:
    """Factory to instantiate the appropriate loader based on filename extension."""

    @staticmethod
    def get_loader(filename: str) -> Tuple[BaseDocumentLoader, DocumentFileType]:
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext == "pdf":
            return PDFLoader(), DocumentFileType.PDF
        elif ext in ("docx", "doc"):
            return DocxLoader(), DocumentFileType.DOCX
        elif ext == "txt":
            return TextLoader(), DocumentFileType.TXT
        elif ext in ("md", "markdown"):
            return TextLoader(), DocumentFileType.MD
        else:
            raise DocumentValidationError(
                f"Unsupported file extension '.{ext}'. Supported: pdf, docx, txt, md"
            )

    @staticmethod
    def calculate_sha256(content: bytes) -> str:
        """Calculate SHA-256 hash of content bytes."""
        return hashlib.sha256(content).hexdigest()
