# Semantic search; returns chunks + metadata + distance/similarity scores.
# Filled in: Module 6
from src.config import MAX_DISTANCE, TOP_K


class Retriever:
    """Finds the chunks that best match a question."""

    def __init__(
        self, embedder, store, top_k: int = TOP_K, max_distance: float = MAX_DISTANCE
    ):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.max_distance = max_distance

    def search(
        self, question: str, top_k: int | None = None, document_id: str | None = None
    ) -> list[dict]:
        question = question.strip()
        if not question:
            return []

        vector = self.embedder.embed_query(question)
        hits = self.store.query(
            vector, n_results=top_k or self.top_k, document_id=document_id
        )

        results = []
        for rank, hit in enumerate(hits, start=1):
            meta = hit["metadata"]
            results.append(
                {
                    "rank": rank,
                    "chunk_id": hit["chunk_id"],
                    "text": hit["text"],
                    "document_id": meta["document_id"],
                    "filename": meta["filename"],
                    "page_number": meta["page_number"],
                    "chunk_index": meta["chunk_index"],
                    "distance": hit["distance"],
                    "similarity": 1 - hit["distance"],
                    "relevant": hit["distance"] <= self.max_distance,
                }
            )
        return results