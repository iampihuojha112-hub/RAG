from src.processing.chunker import chunk_document, chunk_text


def count_tokens(text):
    return len(text.split())


def sentences(n):
    return " ".join(f"This is sentence number {i} in the test text." for i in range(n))


def test_chunks_stay_under_the_token_limit():
    chunks = chunk_text(sentences(100), count_tokens, max_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 50 for c in chunks)


def test_neighbouring_chunks_overlap():
    chunks = chunk_text(sentences(100), count_tokens, max_tokens=50, overlap_tokens=10)
    last_sentence = chunks[0][chunks[0].rindex("This is sentence"):]
    assert chunks[1].startswith(last_sentence)


def test_a_very_long_sentence_is_cut_to_fit():
    text = "word " * 300
    chunks = chunk_text(text, count_tokens, max_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 50 for c in chunks)


def test_empty_text_gives_no_chunks():
    assert chunk_text("", count_tokens) == []
    assert chunk_text("   \n\n  ", count_tokens) == []


def make_doc():
    return {
        "document_id": "D1",
        "filename": "f.pdf",
        "file_type": "pdf",
        "file_hash": "h",
        "pages": [
            {"page_number": 1, "text": sentences(30)},
            {"page_number": 2, "text": "PAGETWO appears only here. " + sentences(30)},
        ],
    }


def test_every_chunk_carries_metadata_and_a_unique_id():
    chunks = chunk_document(make_doc(), count_tokens, max_tokens=50, overlap_tokens=10)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c["chunk_id"].startswith("D1_")
        assert c["metadata"]["document_id"] == "D1"
        assert c["metadata"]["filename"] == "f.pdf"
        assert c["metadata"]["file_hash"] == "h"


def test_chunks_never_cross_pages():
    chunks = chunk_document(make_doc(), count_tokens, max_tokens=50, overlap_tokens=10)
    pages = [c["metadata"]["page_number"] for c in chunks]
    assert pages == sorted(pages)
    assert set(pages) == {1, 2}
    assert not any(
        "PAGETWO" in c["text"] for c in chunks if c["metadata"]["page_number"] == 1
    )
    assert any("PAGETWO" in c["text"] for c in chunks if c["metadata"]["page_number"] == 2)