import pytest

from src.llm.citations import extract_cited_numbers


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("Git is distributed [1].", {1}),
        ("A [1] and B [2, 3].", {1, 2, 3}),
        ("Two in a row [1][4].", {1, 4}),
        ("Compact [1,2].", {1, 2}),
        ("No citations here.", set()),
        ("Not a number [x].", set()),
    ],
)
def test_cited_numbers_are_found(answer, expected):
    assert extract_cited_numbers(answer) == expected