import time
from pathlib import Path

from src.core.exceptions import IngestionError
from src.core.logger import get_logger, log_event, timed_ms
from src.ingestion.chunker import DocumentChunker
from src.ingestion.file_registry import FileRegistry
from src.ingestion.pdf_loader import PDFLoader

logger = get_logger(__name__)


class IngestionService:
    """Coordinate PDF extraction, chunking, embedding, and vector storage."""

    def __init__(self, loader: PDFLoader, chunker: DocumentChunker, registry: FileRegistry, vector_store: object) -> None:
        self.loader = loader
        self.chunker = chunker
        self.registry = registry
        self.vector_store = vector_store

    def ingest(self, path: str | Path) -> dict[str, object]:
        start = time.perf_counter()
        file_hash = self.registry.hash_file(path)
        if file_hash in {str(file["file_hash"]) for file in self.vector_store.list_files()}:
            return {"file_name": Path(path).name, "file_hash": file_hash, "duplicate": True, "pages": 0, "chunks": 0}

        pages = self.loader.load(path, file_hash=file_hash)
        chunks = self.chunker.chunk(pages)
        if not chunks:
            raise IngestionError(f"No selectable text found in {Path(path).name}; scanned PDFs need OCR.")

        try:
            self.vector_store.add_chunks(chunks)
        except Exception as error:
            raise IngestionError(f"Could not index {Path(path).name}: {error}") from error

        self.registry.register(path, len(pages), len(chunks))
        data = {
            "file_name": Path(path).name,
            "file_hash": file_hash,
            "duplicate": False,
            "pages": len(pages),
            "chunks": len(chunks),
            "scanned": Path(path).name in self.loader.scanned_files,
        }
        log_event(logger, "ingestion_done", duration_ms=timed_ms(start), data=data)
        return data