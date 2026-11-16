import sys

from src.extraction.pdf_extractor import extract_pdf
from src.processing.cleaner import clean_pages

pages = extract_pdf(sys.argv[1])
cleaned = clean_pages(pages)

index = 1 if len(pages) > 1 else 0
print("=== BEFORE (page", index + 1, ") ===")
print(repr(pages[index]["text"][:500]))
print("=== AFTER ===")
print(repr(cleaned[index]["text"][:500]))

before = sum(len(p["text"]) for p in pages)
after = sum(len(p["text"]) for p in cleaned)
print("Total chars before:", before, "after:", after)