from pathlib import Path

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE_MB = 20

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = str(BASE_DIR / "data" / "chroma_db" / "rag")
COLLECTION_NAME = "documents"
TOP_K = 5
MAX_DISTANCE = 0.75