import re

import numpy as np
import pytest

from src.config import EMBEDDING_PREFIXES, collection_name_for
from src.embeddings.embedder import Embedder
from src.pipeline.rag_pipeline import RAGPipeline
from tests.conftest import FakeEmbedder, FakeLLM


class RecordingModel:
    """Stands in for a SentenceTransformer and remembers what it was asked to encode."""

    def __init__(self):
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.append(list(texts))
        return np.ones((len(texts), 3))


def test_the_original_model_keeps_the_original_collection_name():
    assert collection_name_for("all-MiniLM-L6-v2") == "documents"


def test_other_models_get_their_own_valid_collection_name():
    name = collection_name_for("BAAI/bge-small-en-v1.5")
    assert name == "documents_BAAI_bge-small-en-v1.5"
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]", name)
    assert name != collection_name_for("all-mpnet-base-v2")


def test_prefixes_are_added_to_queries_only_when_configured():
    model = RecordingModel()
    embedder = Embedder("some-model", query_prefix="query: ", document_prefix="passage: ", model=model)

    embedder.embed_query("what is git")
    embedder.embed_texts(["Git is a tool"])

    assert model.seen[0] == ["query: what is git"]
    assert model.seen[1] == ["passage: Git is a tool"]


def test_models_without_prefixes_send_text_unchanged():
    model = RecordingModel()
    embedder = Embedder("all-MiniLM-L6-v2", model=model)
    embedder.embed_query("what is git")
    embedder.embed_texts(["Git is a tool"])
    assert model.seen == [["what is git"], ["Git is a tool"]]


def test_bge_models_use_the_query_instruction_but_not_for_documents():
    for name in ("BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"):
        assert EMBEDDING_PREFIXES[name]["query"].startswith("Represent this sentence")
        assert EMBEDDING_PREFIXES[name]["document"] == ""


def test_prefixes_can_be_turned_off():
    model = RecordingModel()
    embedder = Embedder("BAAI/bge-small-en-v1.5", query_prefix="", model=model)
    embedder.embed_query("what is git")
    assert model.seen == [["what is git"]]


def test_empty_input_needs_no_model_call():
    model = RecordingModel()
    assert Embedder("x", model=model).embed_texts([]) == []
    assert model.seen == []


def test_pipeline_refuses_chunks_longer_than_the_model_can_read(store):
    class SmallModel(FakeEmbedder):
        max_tokens = 100

    with pytest.raises(ValueError, match="too big"):
        RAGPipeline(embedder=SmallModel(), store=store, llm=FakeLLM())


def test_pipeline_accepts_chunks_that_fit(store):
    class BigModel(FakeEmbedder):
        max_tokens = 512

    RAGPipeline(embedder=BigModel(), store=store, llm=FakeLLM())