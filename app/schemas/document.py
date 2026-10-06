from enum import Enum
from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    WEB = "web"
    CODE = "code"


class DocumentMetadata(BaseModel):
    document_id: str = Field(..., description="Unique ID for the raw document")
    title: str
    source_type: SourceType
    source_url_or_path: str
    author: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ParentChunk(BaseModel):
    parent_id: str = Field(..., description="Unique ID for this parent chunk")
    document_id: str
    text: str
    token_count: int
    metadata: DocumentMetadata


class ChildChunk(BaseModel):
    child_id: str = Field(..., description="Unique ID for search snippet")
    parent_id: str = Field(..., description="ID pointing back to parent chunk")
    document_id: str
    text: str
    token_count: int
    page_number: Optional[int] = None
    metadata: DocumentMetadata