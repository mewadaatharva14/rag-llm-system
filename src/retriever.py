"""
Retriever
==========
Embeds the user query and searches Pinecone for the
most similar chunks. Returns top-k chunks with cosine scores.

The retriever is intentionally kept separate from the vector store
because retrieval logic (query preprocessing, top-k selection) is
independent of storage logic (upsert, delete, index management).
"""

from typing import List, Dict
from src.embedder import Embedder
from src.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, config: dict):
        self.embedder     = embedder
        self.vector_store = vector_store
        self.top_k        = config["retrieval"]["top_k"]

    def retrieve(self, query: str) -> List[Dict]:
        """
        Embed query → search Pinecone → return top-k chunks with scores.

        Args:
            query: user question string

        Returns:
            List of dicts: {text, source, doc_type, page_number, score, chunk_id}
            Sorted by score descending — highest similarity first.
        """
        query_embedding = self.embedder.embed_query(query)
        chunks          = self.vector_store.search(query_embedding, top_k=self.top_k)
        chunks.sort(key=lambda x: x["score"], reverse=True)
        return chunks

    def get_top_score(self, chunks: List[Dict]) -> float:
        """
        Return the highest cosine similarity score from retrieved chunks.
        This is the primary signal for the confidence router.
        Returns 0.0 if no chunks retrieved.
        """
        if not chunks:
            return 0.0
        return chunks[0]["score"]