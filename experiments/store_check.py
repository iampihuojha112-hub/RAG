import sys

from src.embeddings.embedder import Embedder
from src.extraction.file_validator import FileValidationError
from src.extraction.loader import load_document
from src.processing.chunker import chunk_document
from src.vectorstore.chroma_store import ChromaStore

if len(sys.argv) < 2:
    raise SystemExit("Usage: add <file> | list | delete <document_id>")

command = sys.argv[1]
store = ChromaStore()

if command == "add":
    try:
        doc = load_document(sys.argv[2])
    except FileValidationError as error:
        raise SystemExit(f"REJECTED: {error}")

    if store.document_exists(doc["file_hash"]):
        raise SystemExit("DUPLICATE: this file is already stored. Nothing added.")

    embedder = Embedder()
    chunks = chunk_document(doc, embedder.count_tokens)
    vectors = embedder.embed_texts([c["text"] for c in chunks])
    store.add_chunks(chunks, vectors)
    print(f"Added {len(chunks)} chunks from {doc['filename']}")
    print("document_id:", doc["document_id"])

elif command == "list":
    for d in store.list_documents():
        print(f"{d['document_id']} | {d['filename']} | {d['chunk_count']} chunks")

elif command == "delete":
    removed = store.delete_document(sys.argv[2])
    print("Chunks removed:", removed)

else:
    raise SystemExit("Unknown command.")

print("Total chunks in database:", store.count())