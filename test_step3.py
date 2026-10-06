from app.schemas.document import DocumentMetadata, SourceType
from app.services.chunker import ParentChildChunker
from app.services.embeddings import EmbeddingService
from app.services.vector_store import QdrantService

# 1. Sample document
text = """
The Apple M-series chips feature unified memory architecture and Metal Performance Shaders (MPS).
PyTorch utilizes MPS acceleration to perform matrix multiplications directly on the Mac GPU.

Qdrant is an open-source vector database designed for high-performance vector similarity search.
It supports filtering by metadata payload and cosine similarity scoring.
"""

metadata = DocumentMetadata(
    document_id="doc_mac_01",
    title="Mac Acceleration & Qdrant",
    source_type=SourceType.MARKDOWN,
    source_url_or_path="/docs/mac.md"
)

# 2. Chunk text
chunker = ParentChildChunker(parent_chunk_size=300, child_chunk_size=100)
parents, children = chunker.process_document(text, metadata)

# 3. Generate embeddings on Mac GPU (MPS)
embedder = EmbeddingService()
child_texts = [c.text for c in children]
vectors = embedder.embed_texts(child_texts)

# 4. Save to Qdrant local DB
qdrant = QdrantService()
qdrant.upsert_child_chunks(children, vectors)

# 5. Search test query
query_text = "How does PyTorch run on Apple Silicon?"
query_vector = embedder.embed_texts([query_text])[0]
search_results = qdrant.search_similar(query_vector, limit=2)

print("\n--- Search Query ---")
print("Query:", query_text)
print("\n--- Top Retrieved Vector Match ---")
print("Similarity Score:", round(search_results[0]["score"], 4))
print("Matched Text Snippet:", search_results[0]["payload"]["text"])
print("Parent ID Link:", search_results[0]["payload"]["parent_id"])