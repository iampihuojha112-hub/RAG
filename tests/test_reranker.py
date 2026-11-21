import pytest

from src.retrieval.factory import make_retriever
from src.retrieval.reranker import Reranker, RerankingRetriever
from src.retrieval.retriever import Retriever
from tests.test_hybrid import add


class FakeCrossEncoder:
    """Scores a pair by how many question words the chunk contains."""

    def predict(self, pairs):
        scores = []
        for question, text in pairs:
            words = set(question.lower().split())
            scores.append(float(len(words & set(text.lower().replace(".", "").split()))))
        return scores


SHORT = "git distributed"
LONG = "git is a distributed version control system used by many teams around the world today"
QUESTION = "git distributed version control system"


def setup(store, embedder):
    add(store, embedder, "A", [SHORT, LONG, "Pasta needs boiling water and salt."])
    base = Retriever(embedder, store)
    return base, Reranker(model=FakeCrossEncoder())


def test_vector_search_alone_puts_the_short_chunk_first(store, embedder):
    base, _ = setup(store, embedder)
    assert base.search(QUESTION)[0]["text"] == SHORT


def test_reranking_can_change_the_order(store, embedder):
    base, reranker = setup(store, embedder)
    results = RerankingRetriever(base, reranker).search(QUESTION, top_k=3)
    assert results[0]["text"] == LONG
    assert results[0]["rank"] == 1
    assert results[0]["previous_rank"] == 2
    assert results[0]["rerank_score"] > results[1]["rerank_score"]


def test_other_fields_survive_reranking(store, embedder):
    base, reranker = setup(store, embedder)
    first = RerankingRetriever(base, reranker).search(QUESTION, top_k=3)[0]
    for key in ("chunk_id", "filename", "page_number", "chunk_index",
                "distance", "similarity", "relevant"):
        assert key in first


def test_only_top_k_are_kept(store, embedder):
    base, reranker = setup(store, embedder)
    assert len(RerankingRetriever(base, reranker).search(QUESTION, top_k=2)) == 2


def test_the_base_retriever_is_asked_for_many_candidates(store, embedder):
    asked = []

    class Spy:
        def search(self, question, top_k=None, document_id=None):
            asked.append(top_k)
            return []

    RerankingRetriever(Spy(), Reranker(model=FakeCrossEncoder()), candidates=20).search("hello", top_k=5)
    assert asked == [20]


def test_blank_question_gives_nothing(store, embedder):
    base, reranker = setup(store, embedder)
    assert RerankingRetriever(base, reranker).search("   ") == []


def test_empty_store_gives_nothing(store, embedder):
    retriever = RerankingRetriever(
        Retriever(embedder, store), Reranker(model=FakeCrossEncoder())
    )
    assert retriever.search(QUESTION) == []


def test_factory_builds_rerank_modes(store, embedder):
    fake = Reranker(model=FakeCrossEncoder())
    assert type(make_retriever("vector+rerank", embedder, store, fake)).__name__ == "RerankingRetriever"
    assert type(make_retriever("hybrid+rerank", embedder, store, fake)).__name__ == "RerankingRetriever"
    with pytest.raises(ValueError):
        make_retriever("vector+magic", embedder, store, fake)
    with pytest.raises(ValueError):
        make_retriever("magic+rerank", embedder, store, fake)