# FastAPI app and endpoints (upload, ask, list, delete, health).
# Filled in: Module 9
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile

from src.api.schemas import (
    AskRequest,
    AskResponse,
    DeleteResponse,
    DocumentInfo,
    UploadResponse,
)
from src.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB
from src.extraction.file_validator import FileValidationError, sanitize_filename
from src.pipeline.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s"
)
logging.getLogger("src").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

READ_BLOCK = 1024 * 1024
MAX_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = RAGPipeline()
    logger.info("RAG pipeline ready")
    yield


app = FastAPI(title="RAG App API", lifespan=lifespan)


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline


@app.get("/health")
def health(pipeline: RAGPipeline = Depends(get_pipeline)):
    return {"status": "ok", "total_chunks": pipeline.store.count()}


@app.post("/upload", response_model=UploadResponse)
def upload(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    filename = sanitize_filename(file.filename or "")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed.")

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / filename
        size = 0
        with open(path, "wb") as out:
            while block := file.file.read(READ_BLOCK):
                size += len(block)
                if size > MAX_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is bigger than {MAX_FILE_SIZE_MB} MB.",
                    )
                out.write(block)

        try:
            return pipeline.ingest(str(path))
        except (FileValidationError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error))


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    try:
        return pipeline.ask(
            payload.question, top_k=payload.top_k, document_id=payload.document_id
        )
    except Exception:
        logger.exception("Answering failed")
        raise HTTPException(
            status_code=500,
            detail="Could not generate an answer. Check the server log.",
        )


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents(pipeline: RAGPipeline = Depends(get_pipeline)):
    return pipeline.list_documents()


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str, pipeline: RAGPipeline = Depends(get_pipeline)):
    removed = pipeline.delete_document(document_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"document_id": document_id, "chunks_removed": removed}