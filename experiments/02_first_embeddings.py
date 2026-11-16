from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

sentences = [
    "A puppy is playing in the garden.",
    "A young dog runs around outside.",
    "The stock market fell sharply today.",
]

embeddings = model.encode(sentences)

print("Type:", type(embeddings))
print("Shape:", embeddings.shape)
print("First 5 numbers of sentence 1:", embeddings[0][:5])
print("Max sequence length:", model.max_seq_length)
print(embeddings.shape)