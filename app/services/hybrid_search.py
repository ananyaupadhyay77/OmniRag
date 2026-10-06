import torch
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class HybridSearchService:
    """Combines BM25 keyword search, dense vector results (RRF), and Cross-Encoder re-ranking."""

    def __init__(self, reranker_model_name: str = "BAAI/bge-reranker-base"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading Cross-Encoder re-ranker model '{reranker_model_name}' on device: {self.device}")
        self.reranker = CrossEncoder(reranker_model_name, device=self.device)

    @staticmethod
    def bm25_search(query: str, corpus_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs sparse keyword matching over chunk texts."""
        tokenized_corpus = [chunk["text"].lower().split() for chunk in corpus_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Sort chunks by BM25 score
        scored_chunks = sorted(zip(scores, corpus_chunks), key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Merges two ranked lists using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        def process_list(results: List[Dict[str, Any]]):
            for rank, item in enumerate(results):
                chunk_id = item["payload"]["child_id"]
                chunk_map[chunk_id] = item["payload"]
                # RRF Formula: 1 / (k + rank_index)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))

        process_list(dense_results)
        process_list(sparse_results)

        # Sort by combined RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        return [chunk_map[cid] for cid in sorted_ids]

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Uses Cross-Encoder to re-score candidate snippets with high precision."""
        if not candidate_chunks:
            return []

        # Build (query, document) pairs for the Cross-Encoder
        pairs = [[query, chunk["text"]] for chunk in candidate_chunks]
        scores = self.reranker.predict(pairs)

        # Attach scores to chunks and sort
        for idx, chunk in enumerate(candidate_chunks):
            chunk["rerank_score"] = float(scores[idx])

        ranked_chunks = sorted(candidate_chunks, key=lambda x: x["rerank_score"], reverse=True)
        return ranked_chunks[:top_k]