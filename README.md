# 🔍 RAG LLM System — Multi-Document QA with Confidence Routing

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.0-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000?style=flat)](https://pinecone.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-00C28B?style=flat)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-mewadaatharva14-181717?style=flat&logo=github)](https://github.com/mewadaatharva14)

> Production-grade RAG system with confidence-based routing — answers questions
> from a mixed knowledge base of PDFs, Wikipedia articles, and ArXiv papers.
> Uses two-stage scoring (cosine similarity + cross-encoder reranker) to decide
> whether to answer from retrieved context or fall back to LLM knowledge.

---

## 📌 Overview

Most RAG systems blindly pass retrieved chunks to the LLM regardless of relevance.
When context is irrelevant, the LLM hallucinates — producing confident wrong answers.

This system solves that with **confidence-based routing**:

```
Question → Retrieve top-k chunks → Cosine score check
  score ≥ 0.75          → RAG answer  (clearly relevant)
  score < 0.50          → LLM answer  (clearly irrelevant)
  score in [0.50, 0.75] → Cross-encoder reranker → final decision
```

---

## 🗂️ Project Structure

```
rag-llm-system/
├── src/
│   ├── document_processor.py  ← PDF + Wikipedia + ArXiv loading + chunking
│   ├── embedder.py            ← BAAI/bge-small-en-v1.5 embeddings
│   ├── vector_store.py        ← Pinecone upsert, search, delete
│   ├── retriever.py           ← query → embed → Pinecone search
│   ├── reranker.py            ← cross-encoder borderline refinement
│   ├── router.py              ← confidence routing logic
│   ├── generator.py           ← flan-t5-base RAG + LLM mode
│   ├── logger.py              ← query logging to JSON
│   └── rag_pipeline.py        ← end-to-end orchestration
│
├── notebooks/
│   └── 01_rag_pipeline_demo.ipynb
│
├── configs/
│   └── rag_config.yaml        ← all hyperparameters
│
├── data/sample_docs/          ← committed sample PDFs
├── logs/                      ← query logs (gitignored)
├── assets/                    ← routing analysis plots
├── .env.example               ← environment variable template
├── requirements.txt
├── app.py                     ← FastAPI entry point
└── README.md
```

---

## 🏗️ Architecture

### Embedding — BAAI/bge-small-en-v1.5

384-dimensional dense embeddings. BGE models use a query prefix for better retrieval:

- Chunks embedded as plain text
- Queries prefixed with `"Represent this sentence for searching relevant passages: "`

This asymmetric encoding improves retrieval quality by ~5-8% over symmetric encoding.

### Vector Database — Pinecone

Each chunk stored with metadata:

```json
{
  "id":       "chunk_id",
  "values":   [0.1, 0.3, ...],
  "metadata": {
    "text":        "chunk text...",
    "source":      "wikipedia:Transformer",
    "doc_type":    "wikipedia",
    "page_number": 1
  }
}
```

### Confidence Routing

$$\text{route} = \begin{cases} \text{RAG} & \text{if } s_{cos} \geq 0.75 \\ \text{LLM} & \text{if } s_{cos} < 0.50 \\ \text{Reranker} \to \text{RAG/LLM} & \text{if } 0.50 \leq s_{cos} < 0.75 \end{cases}$$

### Reranker — cross-encoder/ms-marco-MiniLM-L-6-v2

Cross-encoder sees both query and chunk simultaneously — captures interaction that
bi-encoder (cosine similarity) misses. Only triggered in the borderline zone
to keep average latency low.

$$s_{rerank} = \sigma(f_{CE}(\text{query}, \text{chunk}))$$

### Generator — google/flan-t5-base

Two prompt modes:

**RAG mode** — answer grounded in retrieved context:
```
Answer the question based on the context below.
Context: <chunk1> <chunk2> ...
Question: <question>
Answer:
```

**LLM mode** — answer from model's own knowledge:
```
Answer the following question:
Question: <question>
Answer:
```

---

## 📊 Results

### Model Comparison

| Query | Route | Cosine Score | Rerank Score | Latency |
|---|---|---|---|---|
| What is self-attention in transformers? | rag | 0.794 | — | 2190ms |
| What is the capital of Australia? | llm | 0.539 | 0.000 | 1243ms |

### Routing Stats (2 queries)
| Metric | Value |
|---|---|
| Total queries | 2 |
| RAG route | 1 (50%) |
| LLM route | 1 (50%) |
| Avg cosine score | 0.667 |
| Avg latency | 1716ms |

### Knowledge Base
| Document | Type | Chunks |
|---|---|---|
| Transformer (deep learning) | Wikipedia | 293 |
| Attention Is All You Need (1706.03762) | ArXiv | 92 |
| Constitution of India | Wikipedia | 187 |
---

## ⚙️ Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/mewadaatharva14/rag-llm-system.git
cd rag-llm-system
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up Pinecone**
```bash
# Sign up at https://www.pinecone.io (free tier)
# Copy your API key
cp .env.example .env
# Edit .env and add your PINECONE_API_KEY
```

**5. Run the API**
```bash
uvicorn app:app --reload --port 8000
```

**6. Ingest documents**
```bash
# Wikipedia article
curl -X POST http://localhost:8000/ingest/url \
     -H "Content-Type: application/json" \
     -d '{"source": "Transformer (machine learning model)", "doc_type": "wikipedia"}'

# ArXiv paper
curl -X POST http://localhost:8000/ingest/url \
     -H "Content-Type: application/json" \
     -d '{"source": "1706.03762", "doc_type": "arxiv"}'

# PDF file
curl -X POST http://localhost:8000/ingest/file \
     -F "file=@data/sample_docs/your_document.pdf"
```

**7. Query**
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the self-attention mechanism?"}'
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ingest/file` | Upload PDF file |
| POST | `/ingest/url` | Ingest Wikipedia or ArXiv |
| POST | `/query` | Ask a question |
| GET | `/documents` | List indexed documents |
| DELETE | `/document` | Remove a document |
| GET | `/health` | Health check + Pinecone status |
| GET | `/logs` | Recent query logs |
| GET | `/stats` | Routing statistics |

### Example Query Response

```json
{
  "answer": "Self-attention allows each token to attend to all other tokens...",
  "route_used": "rag",
  "routing_reason": "Cosine score 0.847 >= threshold 0.75 — using RAG answer",
  "cosine_score": 0.847,
  "rerank_score": null,
  "source_chunks": [...],
  "source_documents": ["wikipedia:Transformer (machine learning model)"],
  "latency_ms": 1243.5
}
```

---

## 🔑 Key Implementation Details

**Why BAAI/bge-small-en-v1.5 over all-MiniLM-L6-v2:**
BGE models are trained with hard negative mining on MS MARCO and BEIR benchmarks.
They outperform MiniLM on retrieval tasks by ~3-5% while remaining fast enough for CPU.

**Why two-stage scoring instead of just reranking everything:**
Cross-encoder reranking is ~10× slower than cosine similarity. Running it on every
query would make the API too slow. The two-stage approach applies the expensive
reranker only where it matters — the borderline zone — keeping average latency low.

**Why RecursiveCharacterTextSplitter for mixed domains:**
Legal text has long paragraphs. Research papers have equations and citations.
Wikipedia has section headers. Fixed-size chunking breaks all of these at wrong
boundaries. Recursive splitter tries `\n\n` first, then `\n`, then `.`, then space —
always splitting at the most natural boundary available.

**Why log_var not var in embeddings:**
Not applicable here — but the same principle applies to all our numerical outputs:
store scores in a range that avoids numerical instability. Cosine scores are
normalized to [0,1] by `normalize_embeddings=True` in sentence-transformers.

---

## 📚 References

| Resource | Link |
|---|---|
| RAG Paper | [Lewis et al. 2020](https://arxiv.org/abs/2005.11401) |
| BGE Embeddings | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| MS-MARCO Reranker | [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |
| Pinecone Docs | [docs.pinecone.io](https://docs.pinecone.io) |
| LangChain Docs | [python.langchain.com](https://python.langchain.com) |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Made with 🧠 by <a href="https://github.com/mewadaatharva14">mewadaatharva14</a>
</p>