"""
Vector Store
=============
Pinecone v5.4+ wrapper — upsert, search, delete, list documents.

Why Pinecone over local FAISS:
- Persistent across restarts — no re-indexing every run
- Managed infrastructure — no server to maintain
- Built-in metadata filtering — filter by doc_type, source
- Real-time upserts — add documents without rebuilding index
- Free tier: 1 index, 100K vectors — enough for this project

Metadata stored per vector:
  text, source, doc_type, page_number, chunk_id
"""

import os
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    def __init__(self, config: dict):
        self.index_name = config["pinecone"]["index_name"]
        self.dimension  = config["pinecone"]["dimension"]
        self.metric     = config["pinecone"]["metric"]

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY not found. "
                "Create a .env file with PINECONE_API_KEY=your_key"
            )

        self.pc    = Pinecone(api_key=api_key)
        self.index = self._get_or_create_index()
        print(f"VectorStore connected: index='{self.index_name}'")

    def _get_or_create_index(self):
        existing = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            print(f"Creating Pinecone index: '{self.index_name}'")
            self.pc.create_index(
                name      = self.index_name,
                dimension = self.dimension,
                metric    = self.metric,
                spec      = ServerlessSpec(
                    cloud  = "aws",
                    region = "us-east-1",
                ),
            )
            # wait for index to be ready
            import time
            while not self.pc.describe_index(self.index_name).status["ready"]:
                print("Waiting for index to be ready...")
                time.sleep(2)

        return self.pc.Index(self.index_name)

    def upsert(self, chunks: List[Dict], embeddings: np.ndarray) -> int:
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id":     chunk["chunk_id"],
                "values": embedding.tolist(),
                "metadata": {
                    "text":        chunk["text"][:1000],
                    "source":      chunk["source"],
                    "doc_type":    chunk["doc_type"],
                    "page_number": chunk.get("page_number", 1),
                },
            })

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

        print(f"Upserted {len(vectors)} vectors to Pinecone")
        return len(vectors)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        results = self.index.query(
            vector           = query_embedding.tolist(),
            top_k            = top_k,
            include_metadata = True,
        )

        chunks = []
        for match in results["matches"]:
            chunks.append({
                "text":        match["metadata"]["text"],
                "source":      match["metadata"]["source"],
                "doc_type":    match["metadata"]["doc_type"],
                "page_number": match["metadata"].get("page_number", 1),
                "score":       match["score"],
                "chunk_id":    match["id"],
            })

        return chunks

    def delete_document(self, source: str) -> None:
        self.index.delete(filter={"source": {"$eq": source}})
        print(f"Deleted all chunks from source: {source}")

    def get_stats(self) -> dict:
        stats = self.index.describe_index_stats()
        return {
            "total_vector_count": stats.total_vector_count,
            "dimension":          stats.dimension,
            "index_fullness":     stats.index_fullness,
        }