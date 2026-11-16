import sys

from src.embeddings.embedder import Embedder
from src.extraction.loader import load_document
from src.processing.chunker import chunk_document

embedder = Embedder()
doc = load_document(sys.argv[1])
chunks = chunk_document(doc, embedder.count_tokens)

sizes = [embedder.count_tokens(c["text"]) for c in chunks]
print("Pages:", len(doc["pages"]), "| Chunks:", len(chunks))
print("Tokens per chunk -> min:", min(sizes), "max:", max(sizes), "avg:", sum(sizes) // len(sizes))
print("Chunks over the limit:", sum(s > embedder.max_tokens - 2 for s in sizes))

for c in chunks[:3]:
    meta = c["metadata"]
    print(f"\n--- chunk {meta['chunk_index']} | {meta['filename']} | page {meta['page_number']} ---")
    print(c["text"][:300])