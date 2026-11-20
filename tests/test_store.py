import pytest


def vectors(n):
    return [[1.0, 0.0, 0.0]] * n


def test_chunks_are_saved_and_counted(store, make_chunks):
    store.add_chunks(make_chunks("A", 3), vectors(3))
    assert store.count() == 3


def test_duplicates_are_found_by_file_hash(store, make_chunks):
    store.add_chunks(make_chunks("A", 2, file_hash="h1", filename="a.pdf"), vectors(2))
    assert store.document_exists("h1")
    assert not store.document_exists("other")
    assert store.find_document_by_hash("h1") == {"document_id": "A", "filename": "a.pdf"}
    assert store.find_document_by_hash("other") is None


def test_delete_removes_only_that_document(store, make_chunks):
    store.add_chunks(make_chunks("A", 3, file_hash="ha"), vectors(3))
    store.add_chunks(make_chunks("B", 2, file_hash="hb"), vectors(2))
    assert store.delete_document("A") == 3
    assert store.count() == 2
    assert [d["document_id"] for d in store.list_documents()] == ["B"]


def test_deleting_an_unknown_document_returns_zero(store):
    assert store.delete_document("nope") == 0


def test_missing_page_number_is_saved_as_zero(store, make_chunks):
    store.add_chunks(make_chunks("A", 1, page=None), vectors(1))
    assert store.collection.get()["metadatas"][0]["page_number"] == 0


def test_mismatched_lengths_are_rejected(store, make_chunks):
    with pytest.raises(ValueError):
        store.add_chunks(make_chunks("A", 2), vectors(1))


def test_big_documents_are_saved_in_batches(store, make_chunks):
    store.add_chunks(make_chunks("A", 1200), vectors(1200))
    assert store.count() == 1200