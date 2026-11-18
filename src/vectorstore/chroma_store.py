# ChromaDB wrapper: add, query, duplicate check (hash), delete by document_id.
# Filled in: Module 5
import chromadb

from src.config import CHROMA_PATH, COLLECTION_NAME

BATCH_SIZE = 500


def _clean_metadata(metadata: dict) -> dict:
    cleaned = dict(metadata)
    if cleaned.get("page_number") is None:
        cleaned["page_number"] = 0
    return cleaned


class ChromaStore:
    """Saves and manages chunks in ChromaDB."""

    def __init__(self, path: str = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def count(self) -> int:
        return self.collection.count()

    def document_exists(self, file_hash: str) -> bool:
        found = self.collection.get(where={"file_hash": file_hash}, limit=1)
        return len(found["ids"]) > 0

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            self.collection.add(
                ids=[c["chunk_id"] for c in batch],
                embeddings=embeddings[start : start + BATCH_SIZE],
                documents=[c["text"] for c in batch],
                metadatas=[_clean_metadata(c["metadata"]) for c in batch],
            )

    def list_documents(self) -> list[dict]:
        data = self.collection.get(include=["metadatas"])
        documents = {}
        for meta in data["metadatas"]:
            doc_id = meta["document_id"]
            if doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta["filename"],
                    "file_type": meta["file_type"],
                    "chunk_count": 0,
                }
            documents[doc_id]["chunk_count"] += 1
        return list(documents.values())

    def delete_document(self, document_id: str) -> int:
        found = self.collection.get(where={"document_id": document_id})
        count = len(found["ids"])
        if count:
            self.collection.delete(where={"document_id": document_id})
        return count

    def query(
        self,
        embedding: list[float],
        n_results: int = 5,
        document_id: str | None = None,
    ) -> list[dict]:
        total = self.count()
        if total == 0:
            return []

        kwargs = {}
        if document_id:
            kwargs["where"] = {"document_id": document_id}

        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, total),
            **kwargs,
        )

        hits = []
        for chunk_id, text, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                {"chunk_id": chunk_id, "text": text, "metadata": meta, "distance": dist}
            )
        return hits