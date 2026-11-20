import hashlib
import math
import re

import pymupdf
import pytest
from docx import Document

from src.llm.prompt_builder import REWRITE_INSTRUCTION
from src.pipeline.rag_pipeline import RAGPipeline
from src.vectorstore.chroma_store import ChromaStore


class FakeEmbedder:
    """Fast stand-in for the real model: word-hash vectors, no downloads."""

    def count_tokens(self, text):
        return len(text.split())

    def _vector(self, text):
        vec = [0.0] * 64
        for word in re.findall(r"[a-z]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_texts(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, query):
        return self._vector(query)


class FakeLLM:
    """Stand-in for Groq: records calls and returns fixed text."""

    def __init__(self):
        self.answer_calls = []
        self.rewrite_calls = []
        self.fail_rewrite = False

    def generate(self, prompt, system_instruction=None):
        if system_instruction == REWRITE_INSTRUCTION:
            self.rewrite_calls.append(prompt)
            if self.fail_rewrite:
                raise RuntimeError("rewrite failed")
            return "what is git distributed version control"
        self.answer_calls.append(prompt)
        return "Git is distributed [1]."


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path):
    return ChromaStore(path=str(tmp_path / "db"))


@pytest.fixture
def pipeline(embedder, store):
    return RAGPipeline(embedder=embedder, store=store, llm=FakeLLM())


@pytest.fixture
def make_pdf(tmp_path):
    def _make(name, pages):
        path = tmp_path / name
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text)
        doc.save(str(path))
        doc.close()
        return str(path)

    return _make


@pytest.fixture
def make_docx(tmp_path):
    def _make(name, paragraphs):
        path = tmp_path / name
        document = Document()
        for text in paragraphs:
            document.add_paragraph(text)
        document.save(str(path))
        return str(path)

    return _make


@pytest.fixture
def make_chunks():
    def _make(document_id, count, file_hash="hash", filename="doc.pdf", page=1):
        return [
            {
                "chunk_id": f"{document_id}_{i}",
                "text": f"chunk {i} of {document_id}",
                "metadata": {
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": "pdf",
                    "file_hash": file_hash,
                    "page_number": page,
                    "chunk_index": i,
                },
            }
            for i in range(count)
        ]

    return _make