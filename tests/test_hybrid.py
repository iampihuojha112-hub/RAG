import pytest

from src.retrieval.factory import make_retriever
from src.retrieval.hybrid_retriever import HybridRetriever


def add(store, embedder, document_id, texts, file_hash=None):
    chunks = [
        {
            "chunk_id": f"{document_id}_{i}",
            "text": text,
            "metadata": {
                "document_id": document_id,
                "filename": f"{document_id}.pdf",
                "file_type": "pdf",
                "file_hash": file_hash or f"hash_{document_id}",
                "page_number": 1,
                "chunk_index": i,
            },
        }
        for i, text in enumerate(texts)
    ]
    store.add_chunks(chunks, embedder.embed_texts(texts))


TEXTS = [
    "Local version control used a simple database on disk.",
    "Centralized version control uses a single server for everyone.",
    "Distributed version control mirrors the whole repository.",
    "The zebracorn protocol is mentioned only in this sentence.",
]


def test_results_have_the_same_shape_as_normal_search(store, embedder):
    add(store, embedder, "A", TEXTS)
    results = HybridRetriever(embedder, store).search("single server version control")
    first = results[0]
    for key in ("rank", "chunk_id", "text", "document_id", "filename", "page_number",
                "chunk_index", "distance", "similarity", "relevant"):
        assert key in first
    assert first["rank"] == 1
    assert first["similarity"] == pytest.approx(1 - first["distance"])


def test_a_rare_exact_word_finds_its_chunk(store, embedder):
    add(store, embedder, "A", TEXTS)
    results = HybridRetriever(embedder, store).search("zebracorn")
    assert "zebracorn" in results[0]["text"]
    assert results[0]["keyword_rank"] == 1


def test_keyword_only_chunks_get_a_correct_distance(store, embedder):
    add(store, embedder, "A", TEXTS)
    retriever = HybridRetriever(embedder, store, candidates=1)
    results = retriever.search("zebracorn version control", top_k=3)

    query_vector = embedder.embed_query("zebracorn version control")
    expected = {h["chunk_id"]: h["distance"] for h in store.query(query_vector, n_results=4)}
    assert any(r["vector_rank"] is None for r in results)
    for r in results:
        assert r["distance"] == pytest.approx(expected[r["chunk_id"]], abs=1e-3)


def test_document_filter_is_respected(store, embedder):
    add(store, embedder, "A", TEXTS)
    add(store, embedder, "B", ["Pasta needs boiling water and salt."])
    results = HybridRetriever(embedder, store).search("version control", document_id="B")
    assert all(r["document_id"] == "B" for r in results)


def test_index_follows_adds_and_deletes(store, embedder):
    retriever = HybridRetriever(embedder, store)
    assert retriever.search("zebracorn") == []

    add(store, embedder, "A", TEXTS)
    assert "zebracorn" in retriever.search("zebracorn")[0]["text"]

    store.delete_document("A")
    assert retriever.search("zebracorn") == []

    add(store, embedder, "B", ["Another zebracorn story."], file_hash="other")
    assert retriever.search("zebracorn")[0]["document_id"] == "B"


def test_blank_question_returns_nothing(store, embedder):
    add(store, embedder, "A", TEXTS)
    assert HybridRetriever(embedder, store).search("   ") == []


def test_factory_picks_the_mode(store, embedder):
    assert type(make_retriever("hybrid", embedder, store)).__name__ == "HybridRetriever"
    assert type(make_retriever("vector", embedder, store)).__name__ == "Retriever"
    with pytest.raises(ValueError):
        make_retriever("magic", embedder, store)