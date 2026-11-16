# Extract paragraphs/tables from DOCX with python-docx.
# Filled in: Module 3
from docx import Document


def extract_docx(path: str) -> list[dict]:
    """Return a single 'page' dict. DOCX has no fixed pages, so page_number is None."""
    doc = Document(path)

    parts = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            parts.append(" | ".join(cells))

    return [{"page_number": None, "text": "\n".join(parts)}]