import pytest

from src.extraction import file_validator
from src.extraction.file_validator import (
    FileValidationError,
    sanitize_filename,
    validate_file,
)


def test_real_pdf_is_accepted(make_pdf):
    validate_file(make_pdf("a.pdf", ["hello world"]))


def test_real_docx_is_accepted(make_docx):
    validate_file(make_docx("a.docx", ["hello world"]))


def test_wrong_extension_is_rejected(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("hello")
    with pytest.raises(FileValidationError, match="not allowed"):
        validate_file(str(path))


def test_fake_pdf_is_rejected(tmp_path):
    path = tmp_path / "fake.pdf"
    path.write_text("hello")
    with pytest.raises(FileValidationError, match="not a real PDF"):
        validate_file(str(path))


def test_fake_docx_is_rejected(tmp_path):
    path = tmp_path / "fake.docx"
    path.write_text("hello")
    with pytest.raises(FileValidationError, match="not a real DOCX"):
        validate_file(str(path))


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    with pytest.raises(FileValidationError, match="empty"):
        validate_file(str(path))


def test_too_big_file_is_rejected(make_pdf, monkeypatch):
    monkeypatch.setattr(file_validator, "MAX_FILE_SIZE_MB", 0.00001)
    with pytest.raises(FileValidationError, match="bigger"):
        validate_file(make_pdf("a.pdf", ["hello world"]))


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileValidationError, match="not found"):
        validate_file(str(tmp_path / "nothing.pdf"))


def test_filenames_are_sanitized():
    assert sanitize_filename("../../evil file(1).pdf") == "evil file_1_.pdf"