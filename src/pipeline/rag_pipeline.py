# ingest(file) and ask(question) tying all modules together.
# Filled in: Module 8
import logging

from src.embeddings.embedder import Embedder
from src.extraction.loader import load_document
from src.llm.groq_client import GroqClient
from src.llm.prompt_builder import NOT_FOUND_MESSAGE, SYSTEM_INSTRUCTION, build_prompt
from src.processing.chunker import chunk_document
from src.retrieval.retriever import Retriever
from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Connects extraction, chunking, embeddings, storage, retrieval and Groq."""

    def __init__(self, embedder=None, store=None, llm=None):
        self.embedder = embedder or Embedder()
        self.store = store or ChromaStore()
        self.llm = llm or GroqClient()
        self.retriever = Retriever(self.embedder, self.store)

    def ingest(self, file_path: str) -> dict:
        doc = load_document(file_path)

        existing = self.store.find_document_by_hash(doc["file_hash"])
        if existing:
            logger.info("Duplicate skipped: %s", doc["filename"])
            return {
                "status": "duplicate",
                "document_id": existing["document_id"],
                "filename": existing["filename"],
                "chunks": 0,
            }

        chunks = chunk_document(doc, self.embedder.count_tokens)
        if not chunks:
            raise ValueError("No chunks were created from this document.")

        vectors = self.embedder.embed_texts([c["text"] for c in chunks])

        try:
            self.store.add_chunks(chunks, vectors)
        except Exception:
            self.store.delete_document(doc["document_id"])
            raise

        logger.info("Ingested %s: %d chunks", doc["filename"], len(chunks))
        return {
            "status": "added",
            "document_id": doc["document_id"],
            "filename": doc["filename"],
            "chunks": len(chunks),
        }

    def ask(
        self, question: str, top_k: int | None = None, document_id: str | None = None
    ) -> dict:
        question = question.strip()
        if not question:
            return {"answer": "Please type a question.", "found": False, "sources": []}

        results = self.retriever.search(question, top_k=top_k, document_id=document_id)
        relevant = [r for r in results if r["relevant"]]
        logger.info("Question retrieved %d chunks, %d relevant", len(results), len(relevant))

        if not relevant:
            return {"answer": NOT_FOUND_MESSAGE, "found": False, "sources": []}

        prompt = build_prompt(question, relevant)
        # The prompt and SYSTEM_INSTRUCTION go to Groq through generate()
        answer = self.llm.generate(prompt, system_instruction=SYSTEM_INSTRUCTION)

        sources = [
            {
                "number": number,
                "filename": r["filename"],
                "page_number": r["page_number"],
                "chunk_index": r["chunk_index"],
                "distance": r["distance"],
                "text": r["text"],
            }
            for number, r in enumerate(relevant, start=1)
        ]
        return {"answer": answer, "found": True, "sources": sources}

    def list_documents(self) -> list[dict]:
        return self.store.list_documents()

    def delete_document(self, document_id: str) -> int:
        removed = self.store.delete_document(document_id)
        logger.info("Deleted document %s: %d chunks", document_id, removed)
        return removed