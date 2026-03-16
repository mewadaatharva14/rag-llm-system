"""
FastAPI Application
====================
REST API for the RAG system.

Endpoints:
  POST /ingest        — upload file or provide URL/topic to index
  POST /query         — ask a question, get answer + sources + scores
  GET  /documents     — list all indexed documents
  DELETE /document    — remove a document from the index
  GET  /health        — health check + Pinecone connection status
  GET  /logs          — return recent query logs
  GET  /stats         — routing stats, avg scores, total queries

Run:
  uvicorn app:app --reload --port 8000

Test:
  curl -X POST http://localhost:8000/query \
       -H "Content-Type: application/json" \
       -d '{"question": "What is transformer architecture?"}'
"""

import time
import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import shutil
import os

from src.rag_pipeline import RAGPipeline


# ── App setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title       = "RAG LLM System",
    description = "Multi-document RAG with confidence-based routing — "
                  "Pinecone + BAAI/bge-small-en-v1.5 + flan-t5-base",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Load pipeline once at startup ─────────────────────────────────────
pipeline = None


@app.on_event("startup")
async def startup_event():
    global pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    print("RAG Pipeline ready.")


# ── Request / Response models ──────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


class IngestURLRequest(BaseModel):
    source:   str              # Wikipedia topic or ArXiv ID
    doc_type: str              # 'wikipedia' or 'arxiv'


class DeleteRequest(BaseModel):
    source: str                # exact source string to delete


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check — confirms API is running and Pinecone is connected."""
    try:
        stats = pipeline.vector_store.get_stats()
        return {
            "status":        "healthy",
            "pinecone":      "connected",
            "total_vectors": stats.get("total_vector_count", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pinecone error: {str(e)}")


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload a PDF file and add it to the Pinecone index.
    Accepts .pdf files only.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code = 400,
            detail      = "Only PDF files supported. Use /ingest/url for Wikipedia and ArXiv.",
        )

    save_path = f"data/sample_docs/{file.filename}"
    os.makedirs("data/sample_docs", exist_ok=True)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = pipeline.ingest(source=save_path, doc_type="pdf")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/url")
def ingest_url(request: IngestURLRequest):
    """
    Ingest a Wikipedia article or ArXiv paper by topic/ID.

    Examples:
      {"source": "Transformer (machine learning)", "doc_type": "wikipedia"}
      {"source": "2305.10403", "doc_type": "arxiv"}
    """
    if request.doc_type not in ("wikipedia", "arxiv"):
        raise HTTPException(
            status_code = 400,
            detail      = "doc_type must be 'wikipedia' or 'arxiv'",
        )

    try:
        result = pipeline.ingest(
            source   = request.source,
            doc_type = request.doc_type,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query(request: QueryRequest):
    """
    Ask a question. Returns answer + routing decision + source chunks.

    Response includes:
      answer           — generated answer
      route_used       — 'rag' | 'llm' | 'reranked'
      routing_reason   — why this route was chosen
      cosine_score     — top retrieval similarity score
      rerank_score     — cross-encoder score (if borderline)
      source_chunks    — top 3 chunks used for context
      source_documents — unique source document names
      latency_ms       — total response time
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = pipeline.query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    """List all documents currently indexed in Pinecone."""
    try:
        stats = pipeline.vector_store.get_stats()
        return {
            "index_stats":    stats,
            "total_vectors":  stats.get("total_vector_count", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document")
def delete_document(request: DeleteRequest):
    """
    Delete all chunks from a specific document.
    Use the exact source string returned by /documents.

    Example:
      {"source": "wikipedia:Transformer (machine learning)"}
      {"source": "arxiv:2305.10403"}
      {"source": "attention_is_all_you_need.pdf"}
    """
    try:
        pipeline.vector_store.delete_document(request.source)
        return {
            "status":  "deleted",
            "source":  request.source,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs")
def get_logs(n: int = 20):
    """Return the n most recent query logs."""
    try:
        logs = pipeline.logger.get_recent(n=n)
        return {
            "count": len(logs),
            "logs":  logs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    """
    Return routing statistics across all queries.
    Shows how often RAG vs LLM route is used,
    average cosine scores, and average latency.
    """
    try:
        return pipeline.logger.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))