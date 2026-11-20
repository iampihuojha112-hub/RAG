from src.retrieval.retriever import Retriever


def test_results_are_ranked_and_scored(pipeline, make_pdf):
    pipeline.ingest(make_pdf("git.pdf", ["Git is a distributed version control system."]))
    pipeline.ingest(make_pdf("cook.pdf", ["Pasta needs boiling water and a pinch of salt."]))
    retriever = Retriever(pipeline.embedder, pipeline.store)

    results = retriever.search("what is git distributed version control", top_k=5)

    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)
    assert results[0]["filename"] == "git.pdf"
    assert results[0]["rank"] == 1
    assert results[0]["similarity"] == 1 - results[0]["distance"]
    assert results[0]["relevant"] is True


def test_unrelated_questions_are_marked_not_relevant(pipeline, make_pdf):
    pipeline.ingest(make_pdf("git.pdf", ["Git is a distributed version control system."]))
    retriever = Retriever(pipeline.embedder, pipeline.store)
    results = retriever.search("quantum lagrangian equation")
    assert results and not any(r["relevant"] for r in results)


def test_blank_question_returns_nothing(pipeline):
    retriever = Retriever(pipeline.embedder, pipeline.store)
    assert retriever.search("   ") == []