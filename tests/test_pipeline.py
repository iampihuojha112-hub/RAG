import shutil

import pytest

from src.extraction.file_validator import FileValidationError
from src.llm.prompt_builder import NOT_FOUND_MESSAGE
from src.pipeline.rag_pipeline import RAGPipeline
from tests.conftest import FakeLLM

GIT_TEXT = "Git is a distributed version control system. Subversion is a centralized system."
COOK_TEXT = "Pasta needs boiling water and a pinch of salt."
GIT_QUESTION = "what is git distributed version control"
HISTORY = [
    {"role": "user", "content": GIT_QUESTION},
    {"role": "assistant", "content": "Git is distributed [1]."},
]


def test_ingest_then_duplicate(pipeline, make_pdf):
    path = make_pdf("notes.pdf", [GIT_TEXT])
    first = pipeline.ingest(path)
    assert first["status"] == "added" and first["chunks"] >= 1
    second = pipeline.ingest(path)
    assert second["status"] == "duplicate"
    assert second["document_id"] == first["document_id"]
    assert pipeline.store.count() == first["chunks"]


def test_a_renamed_copy_is_still_a_duplicate(pipeline, make_pdf, tmp_path):
    path = make_pdf("notes.pdf", [GIT_TEXT])
    copy = tmp_path / "renamed.pdf"
    shutil.copy(path, copy)
    pipeline.ingest(path)
    assert pipeline.ingest(str(copy))["status"] == "duplicate"


def test_a_fake_pdf_is_rejected_and_nothing_is_stored(pipeline, tmp_path):
    fake = tmp_path / "fake.pdf"
    fake.write_text("hello")
    with pytest.raises(FileValidationError):
        pipeline.ingest(str(fake))
    assert pipeline.store.count() == 0


def test_answers_come_with_sources_and_metadata(pipeline, make_pdf):
    pipeline.ingest(make_pdf("notes.pdf", [GIT_TEXT]))
    result = pipeline.ask(GIT_QUESTION)
    assert result["found"] is True
    source = result["sources"][0]
    assert source["filename"] == "notes.pdf"
    assert source["page_number"] == 1
    assert source["number"] == 1
    assert source["cited"] is True


def test_unrelated_questions_are_refused_without_calling_the_llm(pipeline, make_pdf):
    pipeline.ingest(make_pdf("notes.pdf", [GIT_TEXT]))
    result = pipeline.ask("quantum lagrangian equation")
    assert result["found"] is False
    assert result["answer"] == NOT_FOUND_MESSAGE
    assert pipeline.llm.answer_calls == []


def test_empty_question_is_handled(pipeline):
    result = pipeline.ask("   ")
    assert result["found"] is False
    assert result["sources"] == []


def test_word_files_have_page_zero(pipeline, make_docx):
    pipeline.ingest(make_docx("notes.docx", [GIT_TEXT]))
    result = pipeline.ask(GIT_QUESTION)
    assert result["sources"][0]["page_number"] == 0


def test_document_filter_limits_the_search(pipeline, make_pdf):
    git = pipeline.ingest(make_pdf("git.pdf", [GIT_TEXT]))
    cook = pipeline.ingest(make_pdf("cook.pdf", [COOK_TEXT]))

    everywhere = pipeline.ask(GIT_QUESTION)
    assert everywhere["sources"][0]["filename"] == "git.pdf"

    only_cooking = pipeline.ask(GIT_QUESTION, document_id=cook["document_id"])
    assert only_cooking["found"] is False

    only_git = pipeline.ask(GIT_QUESTION, document_id=git["document_id"])
    assert only_git["found"] is True


def test_deleting_a_document_removes_its_knowledge(pipeline, make_pdf):
    info = pipeline.ingest(make_pdf("notes.pdf", [GIT_TEXT]))
    assert pipeline.delete_document(info["document_id"]) == info["chunks"]
    assert pipeline.store.count() == 0
    assert pipeline.ask(GIT_QUESTION)["found"] is False


def test_a_failed_save_leaves_no_partial_document(embedder, store, make_pdf, monkeypatch):
    pipeline = RAGPipeline(embedder=embedder, store=store, llm=FakeLLM())
    original = store.add_chunks

    def broken(chunks, embeddings):
        original(chunks[:1], embeddings[:1])
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "add_chunks", broken)
    with pytest.raises(RuntimeError):
        pipeline.ingest(make_pdf("notes.pdf", [GIT_TEXT]))
    assert store.count() == 0


def test_follow_up_questions_are_rewritten(pipeline, make_pdf):
    pipeline.ingest(make_pdf("notes.pdf", [GIT_TEXT]))
    result = pipeline.ask("explain that more simply", history=HISTORY)
    assert result["search_query"] == GIT_QUESTION
    assert result["found"] is True
    assert len(pipeline.llm.rewrite_calls) == 1


def test_no_history_means_no_rewrite_call(pipeline, make_pdf):
    pipeline.ingest(make_pdf("notes.pdf", [GIT_TEXT]))
    pipeline.ask(GIT_QUESTION)
    assert pipeline.llm.rewrite_calls == []


def test_a_failed_rewrite_falls_back_to_the_original_question(pipeline, make_pdf):
    pipeline.ingest(make_pdf("notes.pdf", [GIT_TEXT]))
    pipeline.llm.fail_rewrite = True
    result = pipeline.ask("git distributed version control", history=HISTORY)
    assert result["search_query"] == "git distributed version control"
    assert result["found"] is True