import sys

from src.embeddings.embedder import Embedder
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_builder import SYSTEM_INSTRUCTION, build_prompt
from src.retrieval.retriever import Retriever
from src.vectorstore.chroma_store import ChromaStore

if len(sys.argv) < 2:
    raise SystemExit('Usage: python -m experiments.rag_check "your question"')

question = " ".join(sys.argv[1:])

store = ChromaStore()
if store.count() == 0:
    raise SystemExit("Database is empty. Add a document first.")

retriever = Retriever(Embedder(), store)
results = retriever.search(question)
relevant = [r for r in results if r["relevant"]]

print(f"Retrieved {len(results)} chunks, {len(relevant)} marked relevant")

if not relevant:
    print("\nANSWER: I could not find this in the uploaded documents.")
    raise SystemExit

prompt = build_prompt(question, relevant)

print("\n===== PROMPT SENT TO GEMINI =====")
print(prompt)
print("===== END OF PROMPT =====\n")

answer = GeminiClient().generate(prompt, system_instruction=SYSTEM_INSTRUCTION)
print("ANSWER:")
print(answer)

print("\nSOURCES:")
for number, r in enumerate(relevant, start=1):
    page = r["page_number"] if r["page_number"] > 0 else "n/a"
    print(f"[{number}] {r['filename']} | page {page} | distance {r['distance']:.3f}")