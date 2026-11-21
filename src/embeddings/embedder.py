from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_PREFIXES


class Embedder:
    """Turns text into embedding vectors using a Sentence Transformers model."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        model=None,
    ):
        self.model_name = model_name
        self.model = model or SentenceTransformer(model_name)

        defaults = EMBEDDING_PREFIXES.get(model_name, {})
        self.query_prefix = defaults.get("query", "") if query_prefix is None else query_prefix
        self.document_prefix = (
            defaults.get("document", "") if document_prefix is None else document_prefix
        )

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    @property
    def max_tokens(self) -> int:
        return self.model.max_seq_length

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def _encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        return self._encode([self.document_prefix + t for t in texts], batch_size)

    def embed_query(self, query: str) -> list[float]:
        return self._encode([self.query_prefix + query], batch_size=1)[0]