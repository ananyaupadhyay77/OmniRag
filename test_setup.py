
import torch
from app.schemas.document import SourceType, DocumentMetadata

print("1. Apple Silicon GPU Available:", torch.backends.mps.is_available())

meta = DocumentMetadata(
    document_id="doc_123",
    title="Test Document",
    source_type=SourceType.PDF,
    source_url_or_path="/test.pdf"
)
print("2. Schema created successfully for:", meta.title)