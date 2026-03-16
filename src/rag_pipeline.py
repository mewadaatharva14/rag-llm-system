"""
RAG Pipeline
=============
Orchestrates all components end to end.

Ingest flow:
  source + doc_type
  → DocumentProcessor (load + chunk)
  → Embedder (embed chunks)
  → VectorStore (upsert to Pinecone)

Query flow:
  question
  → Retriever (embed + search Pinecone)
  → Router Stage 1 (cosine score check)
  → Reranker if borderline (cross-encoder score)
  → Router Stage 2 (rerank score check)
  → Generator (RAG or LLM mode)
  → QueryLogger (log everything)
  → return full response dict
"""

import time
import yaml
from src.document_processor import DocumentProcessor
from src.embedder            import Embedder
from src.vector_store        import VectorStore
from src.retriever           import Retriever
from src.reranker            import Reranker
from src.router              import Router
from src.generator           import Generator
from src.logger              import QueryLogger
from typing import Dict


def load_config(config_path: str = "configs/rag_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class RAGPipeline:
    def __init__(self, config_path: str = "configs/rag_config.yaml"):
        self.config    = load_config(config_path)

        # initialize all components
        self.processor   = DocumentProcessor(self.config)
        self.embedder    = Embedder(self.config)
        self.vector_store = VectorStore(self.config)
        self.retriever   = Retriever(self.embedder, self.vector_store, self.config)
        self.reranker    = Reranker(self.config)
        self.router      = Router(self.config)
        self.generator   = Generator(self.config)
        self.logger      = QueryLogger(self.config)

        print("\nRAG Pipeline ready.")

    def ingest(self, source: str, doc_type: str) -> Dict:
        """
        Load, chunk, embed and store a document in Pinecone.

        Args:
            source   : file path, Wikipedia topic, or ArXiv ID
            doc_type : 'pdf' | 'wikipedia' | 'arxiv'

        Returns:
            {status, source, doc_type, chunks_added}
        """
        chunks     = self.processor.process(source, doc_type)
        texts      = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_chunks(texts)
        n_upserted = self.vector_store.upsert(chunks, embeddings)

        return {
            "status":       "success",
            "source":       source,
            "doc_type":     doc_type,
            "chunks_added": n_upserted,
        }

    def query(self, question: str) -> Dict:
        """
        Answer a question using the full RAG pipeline.

        Returns:
            {
              answer, route_used, routing_reason,
              cosine_score, rerank_score,
              source_chunks, source_documents,
              latency_ms
            }
        """
        start_time = time.time()

        # Step 1 — retrieve
        chunks          = self.retriever.retrieve(question)
        top_cosine      = self.retriever.get_top_score(chunks)
        top_rerank      = None

        # Step 2 — router stage 1 (cosine only)
        route, reason = self.router.route(
            chunks, top_cosine, top_rerank_score=None
        )

        # Step 3 — rerank if borderline
        borderline = (
            self.router.borderline_lower
            <= top_cosine
            < self.router.cosine_threshold
        )

        if borderline:
            chunks      = self.reranker.rerank(question, chunks)
            top_rerank  = self.reranker.get_top_rerank_score(chunks)
            route, reason = self.router.route(
                chunks, top_cosine, top_rerank_score=top_rerank
            )

        # Step 4 — generate
        answer = self.generator.generate(
            query  = question,
            chunks = chunks if route in ("rag", "reranked") else [],
            mode   = route,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Step 5 — log
        self.logger.log(
            question     = question,
            answer       = answer,
            route        = route,
            reason       = reason,
            cosine_score = top_cosine,
            rerank_score = top_rerank,
            source_chunks = chunks,
            latency_ms   = latency_ms,
        )

        return {
            "answer":           answer,
            "route_used":       route,
            "routing_reason":   reason,
            "cosine_score":     round(top_cosine, 4),
            "rerank_score":     round(top_rerank, 4) if top_rerank else None,
            "source_chunks":    chunks[:3],
            "source_documents": list({c["source"] for c in chunks}),
            "latency_ms":       round(latency_ms, 2),
        }