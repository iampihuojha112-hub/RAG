import sys

from src.extraction.file_validator import FileValidationError
from src.extraction.loader import load_document

try:
    doc = load_document(sys.argv[1])
except FileValidationError as error:
    print("REJECTED:", error)
    raise SystemExit(1)

print("document_id:", doc["document_id"])
print("filename:   ", doc["filename"])
print("file_type:  ", doc["file_type"])
print("file_hash:  ", doc["file_hash"])
print("pages:      ", len(doc["pages"]))