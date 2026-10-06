import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.schemas.document import ChildChunk


class QdrantService:
    """Manages local vector storage and semantic search using Qdrant."""

    def __init__(self, collection_name: str = "omnirag_chunks", storage_path: str = "./qdrant_db"):
        # Local persistent disk storage
        self.client = QdrantClient(path=storage_path)
        self.collection_name = collection_name
        self.vector_size = 384  # Matches BAAI/bge-small-en-v1.5 vector size
        self._init_collection()

    def _init_collection(self):
        """Creates the Qdrant collection if it does not exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            print(f"Created Qdrant collection: {self.collection_name}")

    def upsert_child_chunks(self, child_chunks: List[ChildChunk], embeddings: List[List[float]]):
        """Stores child vectors and payload metadata into Qdrant."""
        points = []
        for chunk, vector in zip(child_chunks, embeddings):
            point_id = str(uuid.uuid4())
            payload = {
                "child_id": chunk.child_id,
                "parent_id": chunk.parent_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "title": chunk.metadata.title,
                "source_type": chunk.metadata.source_type,
                "source_url_or_path": chunk.metadata.source_url_or_path
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Upserted {len(points)} vectors to Qdrant.")

    def search_similar(self, query_vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """Searches Qdrant for the most similar child chunks."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        return [{"score": res.score, "payload": res.payload} for res in results.points]