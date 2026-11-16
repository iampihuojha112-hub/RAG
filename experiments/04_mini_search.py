from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

corpus = [
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

corpus_vecs = model.encode(corpus)

def search(query, top_k=3):
    query_vec = model.encode(query)
    scores = util.cos_sim(query_vec, corpus_vecs)[0]
    ranked = scores.argsort(descending=True)
    print(f"\nQuery: {query}")
    for rank, idx in enumerate(ranked[:top_k], start=1):
        print(f"{rank}. score={float(scores[idx]):.3f}  {corpus[idx]}")

search("How do I keep my pet healthy?")
search("Which language should I use for AI?")