"""
Confidence Router
==================
The core decision logic of this RAG system.

Most RAG systems blindly pass retrieved chunks to the LLM regardless
of how relevant they are. This causes hallucination — the LLM tries
to answer using irrelevant context and produces confident wrong answers.

This router solves that with two-stage confidence scoring:

  Stage 1 — cosine similarity score from Pinecone:
    score >= cosine_threshold (0.75) → RAG answer  (clearly relevant)
    score <  borderline_lower  (0.50) → LLM answer  (clearly irrelevant)
    score in [0.50, 0.75]      → borderline → Stage 2

  Stage 2 — cross-encoder rerank score:
    rerank_score >= rerank_threshold (0.60) → RAG answer
    rerank_score <  rerank_threshold (0.60) → LLM answer

Route enum:
  "rag"      → answer from retrieved context
  "llm"      → answer from LLM's own knowledge (no context)
  "reranked" → borderline case resolved by reranker → RAG answer
"""

from typing import List, Dict, Tuple


class Router:
    def __init__(self, config: dict):
        self.cosine_threshold  = config["retrieval"]["cosine_threshold"]
        self.borderline_lower  = config["retrieval"]["borderline_lower"]
        self.rerank_threshold  = config["retrieval"]["rerank_threshold"]

    def route(
        self,
        chunks: List[Dict],
        top_cosine_score: float,
        top_rerank_score: float = None,
    ) -> Tuple[str, str]:
        """
        Decide routing based on confidence scores.

        Args:
            chunks           : retrieved chunks from Pinecone
            top_cosine_score : highest cosine similarity score
            top_rerank_score : highest rerank score (None if not computed)

        Returns:
            Tuple of (route, reason):
              route  : 'rag' | 'llm' | 'reranked'
              reason : human-readable explanation of decision
        """

        # Stage 1 — cosine score check
        if top_cosine_score >= self.cosine_threshold:
            return (
                "rag",
                f"Cosine score {top_cosine_score:.3f} >= threshold "
                f"{self.cosine_threshold} — using RAG answer",
            )

        if top_cosine_score < self.borderline_lower:
            return (
                "llm",
                f"Cosine score {top_cosine_score:.3f} < lower bound "
                f"{self.borderline_lower} — no relevant context found, "
                f"using LLM knowledge",
            )

        # Stage 2 — borderline zone, check rerank score
        if top_rerank_score is None:
            return (
                "llm",
                f"Cosine score {top_cosine_score:.3f} in borderline zone "
                f"but reranker not available — defaulting to LLM",
            )

        if top_rerank_score >= self.rerank_threshold:
            return (
                "reranked",
                f"Cosine score {top_cosine_score:.3f} borderline but "
                f"rerank score {top_rerank_score:.3f} >= {self.rerank_threshold} "
                f"— using RAG answer after reranking",
            )

        return (
            "llm",
            f"Cosine score {top_cosine_score:.3f} borderline and "
            f"rerank score {top_rerank_score:.3f} < {self.rerank_threshold} "
            f"— context not relevant enough, using LLM knowledge",
        )