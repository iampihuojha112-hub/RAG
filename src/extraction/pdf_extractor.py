# Extract text per page from PDFs with PyMuPDF; keep page numbers.
# Filled in: Module 3
import pymupdf


def extract_pdf(path: str) -> list[dict]:
    """Return one dict per page: {"page_number": int, "text": str}."""
    pages = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            pages.append({"page_number": index, "text": text})
    return pages