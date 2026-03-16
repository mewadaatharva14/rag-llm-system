"""
Reranker
=========
Cross-encoder reranker for borderline retrieval scores.

Why two-stage scoring:
  Stage 1 — cosine similarity (fast, approximate)
    Retrieves top-k candidates from Pinecone in milliseconds.
    Good for clear matches (score > 0.75) and clear misses (< 0.50).
    Unreliable in the 0.50-0.75 borderline zone.

  Stage 2 — cross-encoder reranker (slower, precise)
    Cross-encoder sees BOTH query and chunk together.
    Bi-encoder (cosine) sees them separately — misses interaction.
    Much more accurate for borderline cases.
    Only runs on borderline queries — keeps latency low.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  Trained on MS MARCO passage ranking — 8.8M query-passage pairs.
  Outputs a relevance score (unbounded) — we sigmoid-normalize to (0,1).
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import numpy as np


class Reranker:
    def __init__(self, config: dict):
        self.model_name = config["reranker"]["model"]
        self.threshold  = config["retrieval"]["rerank_threshold"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        print(f"Reranker loaded: {self.model_name}")

    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Score each chunk against the query using cross-encoder.
        Returns chunks sorted by rerank score descending.
        Adds 'rerank_score' key to each chunk dict.

        Args:
            query  : user question
            chunks : list of retrieved chunks from Pinecone

        Returns:
            Same chunks with rerank_score added, sorted best first.
        """
        if not chunks:
            return chunks

        scores = []
        for chunk in chunks:
            inputs = self.tokenizer(
                query,
                chunk["text"],
                return_tensors = "pt",
                truncation     = True,
                max_length     = 512,
                padding        = True,
            )

            with torch.no_grad():
                logits = self.model(**inputs).logits
                # sigmoid normalizes unbounded logit to (0, 1)
                score  = torch.sigmoid(logits[0][0]).item()

            scores.append(score)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = round(score, 4)

        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return chunks

    def get_top_rerank_score(self, chunks: List[Dict]) -> float:
        """
        Return highest rerank score from reranked chunks.
        Returns 0.0 if no chunks or rerank not yet run.
        """
        if not chunks:
            return 0.0
        return chunks[0].get("rerank_score", 0.0)