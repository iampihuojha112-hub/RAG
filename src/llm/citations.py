
import re


def extract_cited_numbers(answer: str) -> set[int]:
    """Find source numbers like [1], [2], 【1】 or 【1, 3】."""
    numbers = set()

    pattern = r"(?:\[|【)(\d+(?:\s*,\s*\d+)*)(?:\]|】)"

    for group in re.findall(pattern, answer):
        numbers.update(int(n) for n in re.findall(r"\d+", group))

    return numbers
