from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker, RerankingRetriever
from src.retrieval.retriever import Retriever


def make_retriever(mode: str, embedder, store, reranker=None):
    """Modes: vector, hybrid, vector+rerank, hybrid+rerank."""
    base_mode, _, extra = mode.partition("+")

    if base_mode == "vector":
        base = Retriever(embedder, store)
    elif base_mode == "hybrid":
        base = HybridRetriever(embedder, store)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    if extra == "":
        return base
    if extra == "rerank":
        return RerankingRetriever(base, reranker or Reranker())
    raise ValueError(f"Unknown retrieval mode: {mode}")