# Picks the right extractor by file type; attaches document_id + metadata.
# Filled in: Module 3
import hashlib
import uuid
from pathlib import Path

from src.extraction.docx_extractor import extract_docx
from src.extraction.file_validator import (
    FileValidationError,
    sanitize_filename,
    validate_file,
)
from src.extraction.pdf_extractor import extract_pdf


def compute_file_hash(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha.update(block)
    return sha.hexdigest()


def load_document(path: str) -> dict:
    validate_file(path)

    file = Path(path)
    extension = file.suffix.lower()

    if extension == ".pdf":
        pages = extract_pdf(path)
    else:
        pages = extract_docx(path)

    total_chars = sum(len(page["text"].strip()) for page in pages)
    if total_chars == 0:
        raise FileValidationError(
            "No readable text found. The file may be a scanned image."
        )

    return {
        "document_id": str(uuid.uuid4()),
        "filename": sanitize_filename(file.name),
        "file_type": extension.lstrip("."),
        "file_hash": compute_file_hash(path),
        "pages": pages,
    }