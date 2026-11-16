import sys

from src.embeddings.embedder import Embedder
from src.extraction.loader import load_document
from src.processing.chunker import split_into_chunks

doc = load_document(sys.argv[1])
text = "\n".join(page["text"] for page in doc["pages"])

chunks = split_into_chunks(text, chunk_size=150, overlap=30)

embedder = Embedder()
tokenizer = embedder.model.tokenizer

print("Total words:", len(text.split()))
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3]):
    words = len(chunk.split())
    tokens = len(tokenizer.encode(chunk))
    print(f"Chunk {i}: {words} words, {tokens} tokens")

if len(chunks) > 1:
    same = chunks[0].split()[-30:] == chunks[1].split()[:30]
    print("Overlap works:", same)

longest = max(len(tokenizer.encode(c)) for c in chunks)
print("Longest chunk (tokens):", longest, "| model limit:", embedder.max_tokens)