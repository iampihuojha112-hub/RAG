import sys

from src.embeddings.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.vectorstore.chroma_store import ChromaStore

if len(sys.argv) < 2:
    raise SystemExit('Usage: python -m experiments.search_check "your question"')

question = " ".join(sys.argv[1:])

store = ChromaStore()
if store.count() == 0:
    raise SystemExit("Database is empty. Add a document first (store_check add ...).")

retriever = Retriever(Embedder(), store)
results = retriever.search(question, top_k=5)

print(f"\nQuestion: {question}")
print(f"Chunks with distance <= {retriever.max_distance} are marked RELEVANT\n")

for r in results:
    page = r["page_number"] if r["page_number"] > 0 else "n/a"
    verdict = "RELEVANT" if r["relevant"] else "TOO FAR"
    print("=" * 70)
    print(
        f"#{r['rank']}  distance={r['distance']:.3f}  "
        f"similarity={r['similarity']:.3f}  [{verdict}]"
    )
    print(f"source: {r['filename']} | page: {page} | chunk: {r['chunk_index']}")
    print("-" * 70)
    print(r["text"])