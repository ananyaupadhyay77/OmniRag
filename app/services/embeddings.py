import torch
from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generates dense vector embeddings using Apple Silicon GPU (MPS)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        # Select Apple GPU if available, else fallback to CPU
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading embedding model '{model_name}' on device: {self.device}")
        
        # BAAI/bge-small-en-v1.5 outputs 384-dimensional vectors
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Converts a list of text strings into vector lists."""
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()