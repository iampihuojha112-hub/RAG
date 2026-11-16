from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

# Part A: similarity between sentences
sentences = [
    "A puppy is playing in the garden.",
    "A young dog runs around outside.",
    "The stock market fell sharply today.",
]
embeddings = model.encode(sentences)
scores = util.cos_sim(embeddings, embeddings)
print("Similarity matrix:")
print(scores)

# Part B: token vectors vs sentence vector
text = "A puppy is playing in the garden."
token_vecs = model.encode(text, output_value="token_embeddings")
sentence_vec = model.encode(text)

print("Token vectors shape:  ", token_vecs.shape)
print("Sentence vector shape:", sentence_vec.shape)

# Part C: verify pooling yourself
manual_vec = token_vecs.mean(dim=0)
print("Manual mean vs real embedding:", util.cos_sim(manual_vec, sentence_vec))