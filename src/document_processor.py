"""
Document Processor
===================
Handles loading and chunking of three document types:
  - PDF files (via PyMuPDF)
  - Wikipedia articles (via wikipedia API)
  - ArXiv research papers (via arxiv API)

All documents are chunked using RecursiveCharacterTextSplitter
which respects paragraph and sentence boundaries — critical for
mixed domains like legal text, research papers, and encyclopedic content.
"""

import os
import arxiv
import fitz
import wikipedia
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict


def load_config(config_path: str = "configs/rag_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DocumentProcessor:
    def __init__(self, config: dict):
        self.chunk_size    = config["chunking"]["chunk_size"]
        self.chunk_overlap = config["chunking"]["chunk_overlap"]
        self.separators    = config["chunking"]["separators"]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size    = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators    = self.separators,
        )

    # ------------------------------------------------------------------
    # PDF loading
    # ------------------------------------------------------------------

    def load_pdf(self, file_path: str) -> List[Dict]:
        """
        Load a PDF file and return list of chunks with metadata.
        Uses PyMuPDF (fitz) — handles scanned + digital PDFs.
        Each chunk carries: text, source, doc_type, page_number, chunk_id
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")

        doc      = fitz.open(file_path)
        full_text_by_page = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                full_text_by_page.append({
                    "text":        text,
                    "page_number": page_num + 1,
                })

        doc.close()

        chunks = []
        for page_data in full_text_by_page:
            page_chunks = self.splitter.split_text(page_data["text"])
            for idx, chunk in enumerate(page_chunks):
                if chunk.strip():
                    chunks.append({
                        "text":        chunk.strip(),
                        "source":      os.path.basename(file_path),
                        "doc_type":    "pdf",
                        "page_number": page_data["page_number"],
                        "chunk_id":    f"{os.path.basename(file_path)}_p{page_data['page_number']}_c{idx}",
                    })

        print(f"PDF loaded: {os.path.basename(file_path)} — {len(chunks)} chunks")
        return chunks

    # ------------------------------------------------------------------
    # Wikipedia loading
    # ------------------------------------------------------------------

    def load_wikipedia(self, topic: str) -> List[Dict]:
        """
        Load a Wikipedia article by topic name.
        wikipedia API returns full article text — we chunk it.
        Handles disambiguation errors gracefully.
        """
        try:
            page  = wikipedia.page(topic, auto_suggest=True)
            title = page.title
            text  = page.content
        except wikipedia.DisambiguationError as e:
            print(f"Disambiguation for '{topic}' — using first option: {e.options[0]}")
            page  = wikipedia.page(e.options[0])
            title = page.title
            text  = page.content
        except wikipedia.PageError:
            raise ValueError(f"Wikipedia page not found: {topic}")

        page_chunks = self.splitter.split_text(text)
        chunks = []
        for idx, chunk in enumerate(page_chunks):
            if chunk.strip():
                chunks.append({
                    "text":        chunk.strip(),
                    "source":      f"wikipedia:{title}",
                    "doc_type":    "wikipedia",
                    "page_number": 1,
                    "chunk_id":    f"wiki_{title.replace(' ', '_')}_c{idx}",
                })

        print(f"Wikipedia loaded: '{title}' — {len(chunks)} chunks")
        return chunks

    # ------------------------------------------------------------------
    # ArXiv loading
    # ------------------------------------------------------------------

    def load_arxiv(self, paper_id: str) -> List[Dict]:
        """
        Load an ArXiv paper by its ID (e.g. '2305.10403').
        Downloads the PDF to data/sample_docs/ then extracts text.
        ArXiv IDs are found in the URL: arxiv.org/abs/2305.10403
        """
        search     = arxiv.Search(id_list=[paper_id])
        results    = list(search.results())

        if not results:
            raise ValueError(f"ArXiv paper not found: {paper_id}")

        paper      = results[0]
        save_path  = f"data/sample_docs/{paper_id.replace('/', '_')}.pdf"

        os.makedirs("data/sample_docs", exist_ok=True)

        if not os.path.exists(save_path):
            print(f"Downloading ArXiv paper: {paper.title}")
            paper.download_pdf(filename=save_path)

        chunks = self.load_pdf(save_path)

        # override metadata with ArXiv-specific info
        for chunk in chunks:
            chunk["source"]   = f"arxiv:{paper_id}"
            chunk["doc_type"] = "arxiv"
            chunk["title"]    = paper.title
            chunk["authors"]  = [str(a) for a in paper.authors]

        print(f"ArXiv loaded: '{paper.title}' — {len(chunks)} chunks")
        return chunks

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def process(self, source: str, doc_type: str) -> List[Dict]:
        """
        Unified entry point for all document types.

        Args:
            source   : file path for PDF, topic name for Wikipedia,
                       paper ID for ArXiv
            doc_type : 'pdf' | 'wikipedia' | 'arxiv'

        Returns:
            List of chunk dicts with text + metadata
        """
        if doc_type == "pdf":
            return self.load_pdf(source)
        elif doc_type == "wikipedia":
            return self.load_wikipedia(source)
        elif doc_type == "arxiv":
            return self.load_arxiv(source)
        else:
            raise ValueError(f"Unknown doc_type: {doc_type}. Use 'pdf', 'wikipedia', or 'arxiv'")