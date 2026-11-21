from src.config import RERANK_CANDIDATES, RERANK_MODEL_NAME, TOP_K


class Reranker:
    """Re-scores (question, chunk) pairs with a cross-encoder model."""

    def __init__(self, model_name: str = RERANK_MODEL_NAME, model=None):
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name)
        self.model = model

    def rerank(self, question: str, results: list[dict], top_k: int) -> list[dict]:
        if not results:
            return []

        scores = self.model.predict([(question, r["text"]) for r in results])
        order = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)

        reranked = []
        for new_rank, index in enumerate(order[:top_k], start=1):
            item = dict(results[index])
            item["previous_rank"] = results[index]["rank"]
            item["rerank_score"] = float(scores[index])
            item["rank"] = new_rank
            reranked.append(item)
        return reranked


class RerankingRetriever:
    """Fetches many candidates with another retriever, then keeps the best after reranking."""

    def __init__(
        self,
        base,
        reranker,
        top_k: int = TOP_K,
        candidates: int = RERANK_CANDIDATES,
    ):
        self.base = base
        self.reranker = reranker
        self.top_k = top_k
        self.candidates = candidates

    def search(
        self, question: str, top_k: int | None = None, document_id: str | None = None
    ) -> list[dict]:
        question = question.strip()
        if not question:
            return []

        top_k = top_k or self.top_k
        pool = self.base.search(
            question, top_k=max(self.candidates, top_k), document_id=document_id
        )
        return self.reranker.rerank(question, pool, top_k)