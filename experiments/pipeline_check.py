
import logging
import sys

from src.extraction.file_validator import FileValidationError
from src.pipeline.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logging.getLogger("src").setLevel(logging.INFO)


if len(sys.argv) < 2:
    raise SystemExit(
        'Usage: add <file> | ask "question" | list | delete <document_id>'
    )

command = sys.argv[1]
pipeline = RAGPipeline()


if command == "add":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: add <file>")

    try:
        result = pipeline.ingest(sys.argv[2])
        print(result)
    except FileValidationError as error:
        raise SystemExit(f"REJECTED: {error}")


elif command == "ask":
    question = " ".join(sys.argv[2:])

    try:
        result = pipeline.ask(question)
    except Exception as error:
        raise SystemExit(f"LLM ERROR: {error}")

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for s in result["sources"]:
        page = (
            s["page_number"]
            if s["page_number"] is not None and s["page_number"] > 0
            else "n/a"
        )

        print(
            f"[{s['number']}] {s['filename']} | "
            f"page {page} | distance {s['distance']:.3f}"
        )


elif command == "list":
    for d in pipeline.list_documents():
        print(
            f"{d['document_id']} | "
            f"{d['filename']} | "
            f"{d['chunk_count']} chunks"
        )


elif command == "delete":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: delete <document_id>")

    print("Chunks removed:", pipeline.delete_document(sys.argv[2]))


else:
    raise SystemExit("Unknown command.")
