import uuid
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.schemas.document import DocumentMetadata, ParentChunk, ChildChunk


class ParentChildChunker:
    """Splits raw text into linked Parent (~1000 chars) and Child (~250 chars) chunks."""

    def __init__(self, parent_chunk_size: int = 1000, child_chunk_size: int = 250):
        # Large splitter for LLM prompt context
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )

        # Small splitter for fine-grained vector retrieval
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=30,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_document(self, text: str, metadata: DocumentMetadata) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        parent_chunks: List[ParentChunk] = []
        child_chunks: List[ChildChunk] = []

        # Step 1: Split raw text into Parent chunks
        raw_parents = self.parent_splitter.split_text(text)

        for raw_parent_text in raw_parents:
            parent_id = str(uuid.uuid4())

            p_chunk = ParentChunk(
                parent_id=parent_id,
                document_id=metadata.document_id,
                text=raw_parent_text,
                token_count=len(raw_parent_text.split()),
                metadata=metadata
            )
            parent_chunks.append(p_chunk)

            # Step 2: Split this Parent chunk into Child chunks
            raw_children = self.child_splitter.split_text(raw_parent_text)

            for raw_child_text in raw_children:
                c_chunk = ChildChunk(
                    child_id=str(uuid.uuid4()),
                    parent_id=parent_id,  # <-- Crucial Link to Parent
                    document_id=metadata.document_id,
                    text=raw_child_text,
                    token_count=len(raw_child_text.split()),
                    metadata=metadata
                )
                child_chunks.append(c_chunk)

        return parent_chunks, child_chunks