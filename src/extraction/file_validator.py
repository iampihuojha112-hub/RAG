# Safe file validation: extension allowlist, size limit, real file type check, filename sanitizing.
# Filled in: Module 3
import re
from pathlib import Path

from src.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB


class FileValidationError(Exception):
    """Raised when a file is not safe or not usable."""


def sanitize_filename(name: str) -> str:
    name = Path(name).name
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name)


def validate_file(path: str) -> None:
    file = Path(path)

    if not file.is_file():
        raise FileValidationError("File not found.")

    extension = file.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise FileValidationError(f"File type {extension} is not allowed.")

    size = file.stat().st_size
    if size == 0:
        raise FileValidationError("File is empty.")
    if size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise FileValidationError(f"File is bigger than {MAX_FILE_SIZE_MB} MB.")

    with open(file, "rb") as f:
        header = f.read(4)

    if extension == ".pdf" and not header.startswith(b"%PDF"):
        raise FileValidationError("This is not a real PDF.")
    if extension == ".docx" and not header.startswith(b"PK"):
        raise FileValidationError("This is not a real DOCX.")