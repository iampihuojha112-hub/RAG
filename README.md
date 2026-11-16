# rag-app

A RAG (Retrieval-Augmented Generation) application built from scratch, module by module.

## Stack
Python, Sentence Transformers (all-MiniLM-L6-v2), ChromaDB, Gemini API, PyMuPDF, python-docx, FastAPI, Streamlit.

## Folder map
- data/raw/          sample PDF/DOCX files for testing
- data/chroma_db/    ChromaDB storage (git-ignored)
- experiments/       throwaway learning scripts
- src/extraction/    file validation + PDF/DOCX text extraction
- src/processing/    cleaning and chunking
- src/embeddings/    embedding model wrapper
- src/vectorstore/   ChromaDB wrapper
- src/retrieval/     semantic search
- src/llm/           Gemini client and prompt builder
- src/pipeline/      full ingest/ask pipeline
- src/api/           FastAPI backend
- frontend/          Streamlit UI
- tests/             tests

## Setup (Windows PowerShell)
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
