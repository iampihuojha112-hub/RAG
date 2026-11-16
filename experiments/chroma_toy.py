import chromadb

from src.embeddings.embedder import Embedder

embedder = Embedder()

client = chromadb.PersistentClient(path="data/chroma_db/toy")
collection = client.get_or_create_collection(
    name="toy",
    configuration={"hnsw": {"space": "cosine"}},
)

sentences = [
    "Dogs need daily walks and plenty of exercise.",
    "Cats often sleep for most of the day.",
    "The new phone has an excellent camera and long battery life.",
    "Python is a popular language for machine learning.",
    "FastAPI makes it easy to build web APIs in Python.",
    "Paris is famous for the Eiffel Tower.",
    "Regular exercise improves heart health.",
    "The recipe needs two cups of flour and one egg.",
    "Neural networks learn patterns from large datasets.",
    "Heavy rain is expected throughout the weekend.",
]

if collection.count() == 0:
    collection.add(
        ids=[f"s{i}" for i in range(len(sentences))],
        embeddings=embedder.embed_texts(sentences),
        documents=sentences,
        metadatas=[{"source": "toy.txt", "line": i} for i in range(len(sentences))],
    )

print("Stored items:", collection.count())

question = "How do I keep my pet healthy?"
result = collection.query(
    query_embeddings=[embedder.embed_query(question)],
    n_results=3,
)

for text, dist, meta in zip(
    result["documents"][0], result["distances"][0], result["metadatas"][0]
):
    print(f"distance={dist:.3f}  similarity={1 - dist:.3f}  line={meta['line']} | {text}")