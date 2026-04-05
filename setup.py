from setuptools import setup, find_packages

setup(
    name         = "rag-llm-system",
    version      = "1.0.0",
    author       = "Atharva Mewada",
    description  = "Production RAG system with confidence-based routing",
    packages     = find_packages(),
    python_requires = ">=3.12",
    install_requires = [
        "langchain==0.3.0",
        "langchain-community==0.3.0",
        "langchain-pinecone==0.2.3",
        "pinecone==5.4.2",
        "sentence-transformers==2.7.0",
        "transformers==4.40.0",
        "torch>=2.1.0",
        "pymupdf==1.24.2",
        "wikipedia==1.4.0",
        "arxiv==2.1.0",
        "fastapi==0.111.0",
        "uvicorn==0.29.0",
        "python-multipart==0.0.9",
        "pyyaml==6.0.1",
        "python-dotenv==1.0.1",
        "tqdm==4.66.2",
        "numpy>=1.24.0",
    ],
)