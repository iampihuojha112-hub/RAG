from evaluation.draft_questions import check_draft, parse_json, pick_chunks

PASSAGE = "Centralized version control uses a single server for everyone on the team."


def test_json_is_found_inside_code_fences_and_extra_text():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 1} Hope it helps.') == {"a": 1}
    assert parse_json("no json here") is None
    assert parse_json("{broken") is None


def test_a_good_draft_passes():
    data = {"question": "Which kind of system keeps everything on one machine?", "evidence": "uses a single server"}
    assert check_draft(data, PASSAGE) is None


def test_bad_drafts_are_rejected():
    ok_question = "Which kind of system keeps everything on one machine?"
    assert "JSON" in check_draft(None, PASSAGE)
    assert "length" in check_draft({"question": "Why?", "evidence": "uses a single server"}, PASSAGE)
    assert "exactly" in check_draft({"question": ok_question, "evidence": "invented words here"}, PASSAGE)
    assert "too easy" in check_draft(
        {"question": "Does it say uses a single server today?", "evidence": "uses a single server"}, PASSAGE
    )
    assert "length" in check_draft({"question": ok_question, "evidence": "single"}, PASSAGE)


def make_chunks(pages, per_page):
    return [
        {"text": "word " * 50, "metadata": {"page_number": p}}
        for p in range(1, pages + 1)
        for _ in range(per_page)
    ]


def test_chunks_are_spread_over_pages_first():
    picked = pick_chunks(make_chunks(pages=5, per_page=4), count=5, seed=1)
    assert sorted(c["metadata"]["page_number"] for c in picked) == [1, 2, 3, 4, 5]


def test_short_chunks_are_ignored_and_count_is_respected():
    chunks = make_chunks(2, 2) + [{"text": "too short", "metadata": {"page_number": 9}}]
    picked = pick_chunks(chunks, count=10, seed=1)
    assert len(picked) == 4
    assert all(c["metadata"]["page_number"] != 9 for c in picked)