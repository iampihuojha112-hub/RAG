# Chunking with overlap; each chunk inherits document metadata.
# Filled in: Module 4
import re
def split_into_chunks(
    text: str, chunk_size: int = 150, overlap: int = 30
) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break

    return chunks

def _cut_long_sentence(sentence, count_tokens, limit):
    n = count_tokens(sentence)
    if n <= limit:
        return [(sentence, n)]

    pieces, current = [], []
    for word in sentence.split():
        candidate = " ".join(current + [word])
        if current and count_tokens(candidate) > limit:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return [(p, count_tokens(p)) for p in pieces]


def _split_into_units(text, count_tokens, limit):
    units = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            units.extend(_cut_long_sentence(sentence, count_tokens, limit))
    return units


def chunk_text(text, count_tokens, max_tokens=200, overlap_tokens=40):
    units = _split_into_units(text, count_tokens, max_tokens - overlap_tokens)

    chunks = []
    current = []
    current_tokens = 0

    for unit, n in units:
        if current and current_tokens + n > max_tokens:
            chunks.append(" ".join(u for u, _ in current))

            overlap, total = [], 0
            for u, k in reversed(current):
                if total + k > overlap_tokens:
                    break
                overlap.insert(0, (u, k))
                total += k
            current, current_tokens = overlap, total

        current.append((unit, n))
        current_tokens += n

    if current:
        chunks.append(" ".join(u for u, _ in current))
    return chunks


def chunk_document(doc, count_tokens, max_tokens=200, overlap_tokens=40):
    all_chunks = []
    chunk_index = 0

    for page in doc["pages"]:
        for text in chunk_text(page["text"], count_tokens, max_tokens, overlap_tokens):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['document_id']}_{chunk_index}",
                    "text": text,
                    "metadata": {
                        "document_id": doc["document_id"],
                        "filename": doc["filename"],
                        "file_type": doc["file_type"],
                        "file_hash": doc["file_hash"],
                        "page_number": page["page_number"],
                        "chunk_index": chunk_index,
                    },
                }
            )
            chunk_index += 1
    return all_chunks