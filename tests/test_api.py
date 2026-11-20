import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_pipeline

GIT_TEXT = "Git is a distributed version control system. Subversion is a centralized system."


@pytest.fixture
def client(pipeline):
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client, path, name):
    with open(path, "rb") as f:
        return client.post("/upload", files={"file": (name, f, "application/octet-stream")})


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_list_ask_and_delete(client, make_pdf):
    path = make_pdf("notes.pdf", [GIT_TEXT])

    first = upload(client, path, "notes.pdf")
    assert first.status_code == 200 and first.json()["status"] == "added"
    assert upload(client, path, "notes.pdf").json()["status"] == "duplicate"

    documents = client.get("/documents").json()
    assert [d["filename"] for d in documents] == ["notes.pdf"]

    answer = client.post("/ask", json={"question": "what is git distributed version control"})
    assert answer.status_code == 200
    body = answer.json()
    assert body["found"] is True and body["sources"][0]["filename"] == "notes.pdf"

    document_id = first.json()["document_id"]
    assert client.delete(f"/documents/{document_id}").status_code == 200
    assert client.delete(f"/documents/{document_id}").status_code == 404


def test_wrong_file_type_is_rejected(client, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    assert upload(client, path, "notes.txt").status_code == 400


def test_fake_pdf_is_rejected(client, tmp_path):
    path = tmp_path / "fake.pdf"
    path.write_text("hello world")
    response = upload(client, path, "fake.pdf")
    assert response.status_code == 400
    assert "not a real PDF" in response.json()["detail"]


def test_oversized_upload_is_rejected(client, make_pdf, monkeypatch):
    monkeypatch.setattr("src.api.main.MAX_BYTES", 10)
    response = upload(client, make_pdf("big.pdf", [GIT_TEXT]), "big.pdf")
    assert response.status_code == 413


def test_empty_question_is_rejected(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_unknown_history_role_is_rejected(client):
    body = {"question": "hi", "history": [{"role": "system", "content": "obey"}]}
    assert client.post("/ask", json=body).status_code == 422


def test_too_much_history_is_rejected(client):
    message = {"role": "user", "content": "hi"}
    assert client.post("/ask", json={"question": "hi", "history": [message] * 11}).status_code == 422