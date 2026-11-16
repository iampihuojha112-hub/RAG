from src.embeddings.embedder import Embedder

embedder = Embedder()
print("Model:", embedder.model_name)
print("Dimension:", embedder.dimension)
print("Max tokens:", embedder.max_tokens)

vecs = embedder.embed_texts(["Hello world", "RAG is useful"])
print("Number of vectors:", len(vecs))
print("Length of each vector:", len(vecs[0]))
print("Type of one vector:", type(vecs[0]))

q = embedder.embed_query("What is RAG?")
print("Query vector length:", len(q))
print("Empty input:", embedder.embed_texts([]))