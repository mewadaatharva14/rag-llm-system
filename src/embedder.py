"""
Embedder
=========
Wraps BAAI/bge-small-en-v1.5 from sentence-transformers.
Embeds both document chunks and user queries.

Why BAAI/bge-small-en-v1.5:
- 384-dim embeddings — compact, fast, matches Pinecone dimension
- Outperforms all-MiniLM-L6-v2 on retrieval benchmarks
- Runs on CPU — no GPU required
- Free, no API key needed

Important: bge models perform better with a query prefix.
Chunks are embedded as-is. Queries are prefixed with
"Represent this sentence for searching relevant passages: "
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class Embedder:
    def __init__(self, config: dict):
        self.model_name = config["embedding"]["model"]
        self.dimension  = config["embedding"]["dimension"]
        self.model      = SentenceTransformer(self.model_name)
        print(f"Embedder loaded: {self.model_name} (dim={self.dimension})")

    def embed_chunks(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of document chunks.
        No prefix — chunks are embedded as plain text.
        Returns numpy array of shape (n_chunks, 384)
        """
        embeddings = self.model.encode(
            texts,
            batch_size           = 32,
            show_progress_bar    = True,
            normalize_embeddings = True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single user query.
        BGE models use a query prefix for better retrieval quality.
        Returns numpy array of shape (384,)
        """
        prefixed_query = (
            f"Represent this sentence for searching relevant passages: {query}"
        )
        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings = True,
        )
        return embedding