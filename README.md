# ⚡ OmniRAG: Multi-Source Production RAG System

A local-first, high-performance Multi-Source Retrieval-Augmented Generation (OmniRAG) platform optimized for Apple Silicon (MPS).

## 🛠️ Architecture & Tech Stack
- **Frontend UI**: Streamlit (Interactive Chat & Citation Viewer)
- **API Backend**: FastAPI + Uvicorn (Async REST Endpoints)
- **Vector Database**: Qdrant (Local persistent store)
- **Dense Embeddings**: `BAAI/bge-small-en-v1.5` (PyTorch `mps` acceleration)
- **Sparse Search**: Rank-BM25
- **Re-ranker**: `BAAI/bge-reranker-base` (Cross-Encoder)
- **Chunking Engine**: Hierarchical Parent-Child Chunking
