import time
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.schemas.query import RAGQueryRequest, RAGQueryResponse, IngestionResponse
from app.schemas.document import DocumentMetadata, SourceType
from app.services.pdf_loader import PDFDocumentLoader
from app.services.chunker import ParentChildChunker
from app.services.embeddings import EmbeddingService
from app.services.vector_store import QdrantService
from app.services.hybrid_search import HybridSearchService
from app.services.llm_engine import LLMEngine

app = FastAPI(
    title="OmniRAG Backend API",
    description="Multi-Source Production RAG Engine with Hybrid Search & Re-ranking",
    version="1.0.0"
)

# Initialize global services on app startup
print("Initializing OmniRAG AI Pipelines...")
chunker = ParentChildChunker()
embedder = EmbeddingService()
qdrant = QdrantService(collection_name="omnirag_api_chunks")
hybrid_search = HybridSearchService()
llm_engine = LLMEngine()


@app.get("/health")
def health_check():
    return {"status": "online", "device": embedder.device}


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """Uploads a PDF file, parses it, chunks it, and indexes vectors into Qdrant."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file temporarily
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        document_id = str(uuid.uuid4())
        full_text, metadata = PDFDocumentLoader.load_pdf(
            file_path=file_path,
            document_id=document_id,
            title=file.filename
        )

        parents, children = chunker.process_document(full_text, metadata)

        # Generate embeddings and store in Qdrant
        child_texts = [c.text for c in children]
        vectors = embedder.embed_texts(child_texts)
        qdrant.upsert_child_chunks(children, vectors)

        return IngestionResponse(
            document_id=document_id,
            filename=file.filename,
            parent_chunks_created=len(parents),
            child_chunks_created=len(children),
            message="Document successfully processed and indexed."
        )

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/query", response_model=RAGQueryResponse)
def query_rag(request: RAGQueryRequest):
    """Executes Hybrid Search + Re-ranking and generates answer prompt with citations."""
    start_time = time.time()

    # 1. Dense Vector Search
    query_vector = embedder.embed_texts([request.query])[0]
    dense_matches = qdrant.search_similar(query_vector, limit=request.top_k * 2)

    # 2. Extract candidates for Re-ranking
    candidate_payloads = [match["payload"] for match in dense_matches]

    # 3. Cross-Encoder Re-ranking
    reranked_chunks = hybrid_search.rerank(request.query, candidate_payloads, top_k=request.top_k)

    # 4. Construct Prompt and Citations
    prompt, citations = llm_engine.build_prompt_and_citations(request.query, reranked_chunks)

    latency = (time.time() - start_time) * 1000

    return RAGQueryResponse(
        query=request.query,
        answer=prompt,  # Contains fully constructed, grounded prompt
        citations=citations,
        retrieval_latency_ms=round(latency, 2)
    )