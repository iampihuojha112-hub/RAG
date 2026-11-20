import sys
from pathlib import Path

from src.config import BASE_DIR
from src.extraction.loader import load_document


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("Usage: python -m evaluation.show_pages <file> [page numbers...]")

    path = Path(argv[0])
    if not path.is_absolute():
        path = BASE_DIR / path
    doc = load_document(str(path))
    wanted = {int(n) for n in argv[1:]}

    for page in doc["pages"]:
        number = page["page_number"] or 0
        if wanted and number not in wanted:
            continue
        print(f"\n===== page {number} =====")
        print(page["text"])


if __name__ == "__main__":
    main()