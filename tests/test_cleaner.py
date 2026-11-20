from src.processing.cleaner import clean_pages, clean_text

WORDS = ["Alpha", "Beta", "Gamma", "Delta"]


def make_pages(count):
    return [
        {
            "page_number": i + 1,
            "text": f"Unit 3 Notes\n{WORDS[i]} is discussed here\nPage {i + 1}",
        }
        for i in range(count)
    ]


def test_hyphenated_words_are_joined():
    assert clean_text("informa-\ntion is useful") == "information is useful"


def test_extra_spaces_and_blank_lines_are_collapsed():
    assert clean_text("a   b\n\n\n\nc") == "a b\n\nc"


def test_repeated_headers_and_page_numbers_are_removed():
    cleaned = clean_pages(make_pages(4))
    for i, page in enumerate(cleaned):
        assert "Unit 3 Notes" not in page["text"]
        assert "Page" not in page["text"]
        assert WORDS[i] in page["text"]
        assert page["page_number"] == i + 1


def test_short_documents_keep_their_headers():
    cleaned = clean_pages(make_pages(2))
    assert "Unit 3 Notes" in cleaned[0]["text"]