# Pydantic request/response models.
# Filled in: Module 9
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_id: str | None = None


class Source(BaseModel):
    number: int
    filename: str
    page_number: int
    chunk_index: int
    distance: float
    text: str


class AskResponse(BaseModel):
    answer: str
    found: bool
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