import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import MAX_DISTANCE, TOP_K

CANDIDATES = 20
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """Combines meaning search (vectors) with keyword search (BM25)."""

    def __init__(
        self,
        embedder,
        store,
        top_k: int = TOP_K,
        max_distance: float = MAX_DISTANCE,
        candidates: int = CANDIDATES,
    ):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.max_distance = max_distance
        self.candidates = candidates
        self._index_key = None
        self._bm25 = None
        self._chunks = []

    def _refresh_index(self) -> None:
        ids = self.store.collection.get(include=[])["ids"]
        key = hash(tuple(sorted(ids)))
        if key == self._index_key:
            return

        data = self.store.collection.get(include=["documents", "metadatas"])
        self._chunks = [
            {"chunk_id": cid, "text": text, "metadata": meta}
            for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
        ]
        self._bm25 = (
            BM25Okapi([tokenize(c["text"]) for c in self._chunks])
            if self._chunks
            else None
        )
        self._index_key = key

    def _keyword_ranks(self, question: str, document_id: str | None) -> dict[str, int]:
        tokens = tokenize(question)
        if not tokens or self._bm25 is None:
            return {}

        scores = self._bm25.get_scores(tokens)
        ranks = {}
        for index in np.argsort(-scores):
            if scores[index] <= 0:
                break
            chunk = self._chunks[index]
            if document_id and chunk["metadata"]["document_id"] != document_id:
                continue
            ranks[chunk["chunk_id"]] = len(ranks) + 1
            if len(ranks) >= self.candidates:
                break
        return ranks

    def _distance_for(self, chunk_ids: list[str], query_vector: list[float]) -> dict[str, float]:
        data = self.store.collection.get(ids=chunk_ids, include=["embeddings"])
        query = np.array(query_vector, dtype=float)
        distances = {}
        for chunk_id, embedding in zip(data["ids"], data["embeddings"]):
            vector = np.array(embedding, dtype=float)
            cosine = float(query @ vector / (np.linalg.norm(query) * np.linalg.norm(vector)))
            distances[chunk_id] = 1 - cosine
        return distances

    def search(
        self, question: str, top_k: int | None = None, document_id: str | None = None
    ) -> list[dict]:
        question = question.strip()
        if not question:
            return []

        top_k = top_k or self.top_k
        self._refresh_index()
        if not self._chunks:
            return []

        query_vector = self.embedder.embed_query(question)

        vector_hits = self.store.query(
            query_vector, n_results=self.candidates, document_id=document_id
        )
        vector_rank = {h["chunk_id"]: r for r, h in enumerate(vector_hits, start=1)}
        distances = {h["chunk_id"]: h["distance"] for h in vector_hits}
        keyword_rank = self._keyword_ranks(question, document_id)

        fused = {}
        for ranks in (vector_rank, keyword_rank):
            for chunk_id, rank in ranks.items():
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        winners = sorted(fused, key=fused.get, reverse=True)[:top_k]

        missing = [cid for cid in winners if cid not in distances]
        if missing:
            distances.update(self._distance_for(missing, query_vector))

        by_id = {c["chunk_id"]: c for c in self._chunks}
        results = []
        for rank, chunk_id in enumerate(winners, start=1):
            chunk = by_id[chunk_id]
            meta = chunk["metadata"]
            distance = distances[chunk_id]
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "document_id": meta["document_id"],
                    "filename": meta["filename"],
                    "page_number": meta["page_number"],
                    "chunk_index": meta["chunk_index"],
                    "distance": distance,
                    "similarity": 1 - distance,
                    "relevant": distance <= self.max_distance,
                    "vector_rank": vector_rank.get(chunk_id),
                    "keyword_rank": keyword_rank.get(chunk_id),
                }
            )
        return results