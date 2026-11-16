from transformers import AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "Antidisestablishmentarianism"

encoded = tokenizer(text)

print("Input IDs:", encoded["input_ids"])
print("Tokens:   ", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
print("Vocabulary size:", tokenizer.vocab_size)
print("Attention Mask:", encoded["attention_mask"])