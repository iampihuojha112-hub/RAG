from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_id: str | None = None
    history: list[ChatMessage] = Field(default_factory=list, max_length=10)


class Source(BaseModel):
    number: int
    filename: str
    page_number: int
    chunk_index: int
    distance: float
    text: str
    cited: bool


class AskResponse(BaseModel):
    answer: str
    found: bool
    search_query: str
    sources: list[Source]


class UploadResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    chunks: int


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_count: int


class DeleteResponse(BaseModel):
    document_id: str
    chunks_removed: int