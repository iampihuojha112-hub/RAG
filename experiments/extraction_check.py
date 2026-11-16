import sys
from pathlib import Path

from src.extraction.pdf_extractor import extract_pdf
from src.extraction.docx_extractor import extract_docx

path = Path(sys.argv[1])
suffix = path.suffix.lower()

if suffix == ".pdf":
    pages = extract_pdf(str(path))
elif suffix == ".docx":
    pages = extract_docx(str(path))
else:
    raise SystemExit(f"Unsupported file type: {suffix}")

print(f"Extracted {len(pages)} page(s) from {path.name}")
for page in pages:
    text = page["text"]
    preview = text[:150].replace("\n", " ")
    print(f"- page {page['page_number']}: {len(text)} chars | {preview}")