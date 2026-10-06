from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.document import SourceType


class IngestionResponse(BaseModel):
    document_id: str
    filename: str
    parent_chunks_created: int
    child_chunks_created: int
    message: str


class RAGQueryRequest(BaseModel):
    query: str = Field(..., example="What causes memory allocation errors on Mac?")
    top_k: int = Field(default=3, ge=1, le=10)


class Citation(BaseModel):
    source_title: str
    source_type: SourceType
    source_path: str
    excerpt: str
    parent_id: str


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    retrieval_latency_ms: float