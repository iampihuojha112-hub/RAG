from pathlib import Path
import re
from pathlib import Path

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE_MB = 20

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = str(BASE_DIR / "data" / "chroma_db" / "rag")
def collection_name_for(model_name: str) -> str:
    """Each embedding model gets its own collection, because vectors from different models can't be compared."""
    if model_name == "all-MiniLM-L6-v2":
        return "documents"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")
    return f"documents_{slug}"


COLLECTION_NAME = collection_name_for(EMBEDDING_MODEL_NAME)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBEDDING_PREFIXES = {
    "BAAI/bge-small-en-v1.5": {"query": BGE_QUERY_PREFIX, "document": ""},
    "BAAI/bge-base-en-v1.5": {"query": BGE_QUERY_PREFIX, "document": ""},
}
TOP_K = 5
MAX_DISTANCE = 0.75

import os

from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_TEMPERATURE = 0.2

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_TEMPERATURE = 0.2

CHUNK_MAX_TOKENS = 200
CHUNK_OVERLAP_TOKENS = 40

RETRIEVAL_MODE = "vector"

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_CANDIDATES = 20