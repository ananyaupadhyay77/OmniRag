from typing import Tuple
from pypdf import PdfReader
from app.schemas.document import DocumentMetadata, SourceType


class PDFDocumentLoader:
    """Loads and extracts text from PDF files."""

    @staticmethod
    def load_pdf(file_path: str, document_id: str, title: str) -> Tuple[str, DocumentMetadata]:
        reader = PdfReader(file_path)
        page_texts = []

        # Extract text page by page with explicit page markers
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                page_texts.append(f"[Page {idx + 1}]\n{text}")

        full_text = "\n\n".join(page_texts)

        # Build schema metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            title=title,
            source_type=SourceType.PDF,
            source_url_or_path=file_path
        )

        return full_text, metadata