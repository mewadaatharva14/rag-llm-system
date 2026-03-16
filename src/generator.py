"""
Generator
==========
flan-t5-base answer generation in two modes:

  RAG mode  — receives retrieved context chunks + question
              generates answer grounded in the documents

  LLM mode  — receives question only
              generates answer from model's own knowledge
              used when retrieval confidence is too low

Why flan-t5-base:
  Instruction-tuned on diverse tasks — handles QA well.
  Runs on CPU — no GPU required.
  Free — no API key.
  Swap model_name in config to upgrade without code changes.

Prompt format for RAG mode:
  "Answer the question based on the context below.
   Context: <chunk1> <chunk2> ...
   Question: <question>
   Answer:"

Prompt format for LLM mode:
  "Answer the following question:
   Question: <question>
   Answer:"
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict


class Generator:
    def __init__(self, config: dict):
        self.model_name     = config["generator"]["model"]
        self.max_new_tokens = config["generator"]["max_new_tokens"]
        self.temperature    = config["generator"]["temperature"]

        print(f"Loading generator: {self.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model     = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.eval()
        print(f"Generator loaded: {self.model_name}")

    def _build_rag_prompt(self, query: str, chunks: List[Dict]) -> str:
        """
        Build RAG prompt — context from retrieved chunks + question.
        Truncates context to avoid exceeding model's max input length.
        """
        context_parts = []
        total_chars   = 0
        max_context   = 1500  # safe limit for flan-t5-base (512 tokens)

        for chunk in chunks:
            if total_chars + len(chunk["text"]) > max_context:
                break
            context_parts.append(chunk["text"])
            total_chars += len(chunk["text"])

        context = "\n\n".join(context_parts)

        return (
            f"Answer the question based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

    def _build_llm_prompt(self, query: str) -> str:
        """
        Build LLM-only prompt — no context, just the question.
        Used when retrieval confidence is too low.
        """
        return (
            f"Answer the following question:\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

    def generate(self, query: str, chunks: List[Dict] = None, mode: str = "rag") -> str:
        """
        Generate answer in RAG or LLM mode.

        Args:
            query  : user question
            chunks : retrieved context chunks (required for RAG mode)
            mode   : 'rag' | 'llm' | 'reranked' (reranked uses RAG prompt)

        Returns:
            Generated answer string
        """
        if mode in ("rag", "reranked") and chunks:
            prompt = self._build_rag_prompt(query, chunks)
        else:
            prompt = self._build_llm_prompt(query)

        inputs = self.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 512,
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens = self.max_new_tokens,
            temperature    = self.temperature,
            do_sample      = self.temperature > 0,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()