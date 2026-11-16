# Text cleaning: whitespace, hyphenation, headers/footers.
# Filled in: Module 3
import re
from collections import Counter


def _normalize(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def remove_repeated_lines(
    pages: list[dict], edge_lines: int = 2, min_share: float = 0.5
) -> list[dict]:
    if len(pages) < 3:
        return pages

    counts = Counter()
    for page in pages:
        lines = [l for l in page["text"].split("\n") if l.strip()]
        edges = {_normalize(l) for l in lines[:edge_lines] + lines[-edge_lines:]}
        counts.update(edges)

    repeated = {line for line, n in counts.items() if n >= len(pages) * min_share}

    result = []
    for page in pages:
        kept = [
            l for l in page["text"].split("\n")
            if not l.strip() or _normalize(l) not in repeated
        ]
        result.append({"page_number": page["page_number"], "text": "\n".join(kept)})
    return result


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pages(pages: list[dict]) -> list[dict]:
    pages = remove_repeated_lines(pages)
    return [
        {"page_number": p["page_number"], "text": clean_text(p["text"])}
        for p in pages
    ]