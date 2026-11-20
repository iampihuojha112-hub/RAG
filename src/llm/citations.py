import re


def extract_cited_numbers(answer: str) -> set[int]:
    """Find source numbers like [1], [2] or [1, 3] inside an answer."""
    numbers = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        numbers.update(int(n) for n in re.findall(r"\d+", group))
    return numbers