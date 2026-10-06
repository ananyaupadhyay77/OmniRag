from app.schemas.document import DocumentMetadata, SourceType
from app.services.chunker import ParentChildChunker
from app.services.embeddings import EmbeddingService
from app.services.vector_store import QdrantService
from app.services.hybrid_search import HybridSearchService

# 1. Setup sample corpus with exact model numbers and general descriptions
text = """
The Apple M3 Max chip contains 16 CPU cores and 40 GPU cores with hardware-accelerated ray tracing.
Error Code ERR_MAC_909 occurs when memory bandwidth overflows in local unified architecture.

FastAPI is a modern web framework for building APIs with Python 3.8+ based on standard Python type hints.
Qdrant vector database allows storing JSON payloads alongside high-dimensional vectors.
"""

metadata = DocumentMetadata(
    document_id="doc_hyb_01",
    title="System Manual & Specs",
    source_type=SourceType.MARKDOWN,
    source_url_or_path="/docs/specs.md"
)

# 2. Chunk & Embed
chunker = ParentChildChunker(parent_chunk_size=300, child_chunk_size=120)
parents, children = chunker.process_document(text, metadata)

embedder = EmbeddingService()
vectors = embedder.embed_texts([c.text for c in children])

qdrant = QdrantService(collection_name="hybrid_test_chunks")
qdrant.upsert_child_chunks(children, vectors)

# 3. Hybrid Search Execution
hybrid_service = HybridSearchService()
query = "What causes error ERR_MAC_909?"

# A. Dense Vector Retrieval
query_vec = embedder.embed_texts([query])[0]
dense_matches = qdrant.search_similar(query_vec, limit=3)

# B. Sparse BM25 Keyword Search
corpus_dicts = [{"child_id": c.child_id, "text": c.text, "payload": {"child_id": c.child_id, "text": c.text, "parent_id": c.parent_id}} for c in children]
sparse_matches = hybrid_service.bm25_search(query, corpus_dicts, top_k=3)

# C. RRF Fusion
candidates = hybrid_service.reciprocal_rank_fusion(dense_matches, sparse_matches)

# D. Cross-Encoder Re-ranking
top_reranked = hybrid_service.rerank(query, candidates, top_k=2)

print("\n--- Search Query ---")
print("Query:", query)
print("\n--- Top Re-ranked Result ---")
print("Re-rank Score:", round(top_reranked[0]["rerank_score"], 4))
print("Snippet:", top_reranked[0]["text"])