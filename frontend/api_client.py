import os

import requests

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")


class ApiError(Exception):
    """A readable error from the backend API."""


def _request(method: str, path: str, timeout: int = 30, **kwargs):
    try:
        response = requests.request(
            method, f"{API_URL}{path}", timeout=timeout, **kwargs
        )
    except requests.ConnectionError:
        raise ApiError("Cannot reach the backend. Is the FastAPI server running?")
    except requests.Timeout:
        raise ApiError("The backend took too long to answer.")

    if response.ok:
        return response.json()

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    if isinstance(detail, list):
        detail = "The request was not valid."
    raise ApiError(str(detail))


def check_health() -> dict:
    return _request("GET", "/health", timeout=5)


def list_documents() -> list[dict]:
    return _request("GET", "/documents")


def upload_file(name: str, data: bytes, content_type: str) -> dict:
    return _request(
        "POST", "/upload", timeout=300, files={"file": (name, data, content_type)}
    )


def ask_question(
    question: str, document_id: str | None = None, top_k: int | None = None
) -> dict:
    body = {"question": question}
    if document_id:
        body["document_id"] = document_id
    if top_k:
        body["top_k"] = top_k
    return _request("POST", "/ask", timeout=120, json=body)


def delete_document(document_id: str) -> dict:
    return _request("DELETE", f"/documents/{document_id}")