from app.schemas.document import DocumentMetadata, SourceType
from app.services.chunker import ParentChildChunker

# Sample document text simulating a multi-paragraph guide
sample_document = """
Retrieval-Augmented Generation (RAG) is an architectural pattern that enhances the accuracy and reliability 
of Large Language Models (LLMs) by fetching relevant facts from external knowledge bases.

In traditional LLM setups, knowledge is frozen at training time. If you query internal company documentation, 
the model cannot provide accurate details and may hallucinate plausible-sounding answers.

By integrating a Vector Database like Qdrant, we store text chunks as mathematical vectors. 
When a user asks a question, we search the vector space for semantically similar text snippets, 
attach them to the LLM prompt, and generate an answer grounded in source documents.
""" * 4

metadata = DocumentMetadata(
    document_id="doc_999",
    title="RAG Overview",
    source_type=SourceType.MARKDOWN,
    source_url_or_path="/docs/rag.md"
)

chunker = ParentChildChunker(parent_chunk_size=400, child_chunk_size=120)
parents, children = chunker.process_document(sample_document, metadata)

print(f"Total Parent Chunks Created: {len(parents)}")
print(f"Total Child Chunks Created: {len(children)}")
print("\n--- Parent Chunk Sample [0] ---")
print("ID:", parents[0].parent_id)
print("Text:", parents[0].text[:100], "...")

print("\n--- Child Chunk Sample [0] ---")
print("ID:", children[0].child_id)
print("Linked Parent ID:", children[0].parent_id)
print("Text:", children[0].text[:100], "...")