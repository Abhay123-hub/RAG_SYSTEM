from pathlib import Path

import pymupdf

from src.core.exceptions import IngestionError
from src.core.logger import get_logger, log_event, timed_ms
from src.models.schemas import PageContent
import time

logger = get_logger(__name__)


class PDFLoader:
    """Extract non-empty, page-numbered text from PDFs."""

    def __init__(self) -> None:
        self.scanned_files: list[str] = []

    def load(self, path: str | Path, file_hash: str = "") -> list[PageContent]:
        start = time.perf_counter()
        file_path = Path(path)
        pages: list[PageContent] = []
        self.scanned_files = [name for name in self.scanned_files if name != file_path.name]
        has_scanned_pages = False
        try:
            with pymupdf.open(file_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    text = page.get_text().strip()
                    if text:
                        pages.append(PageContent(file_name=file_path.name, page=page_number, text=text, file_hash=file_hash))
                    else:
                        has_scanned_pages = True
        except Exception as error:
            raise IngestionError(f"Could not read {file_path.name}: {error}") from error

        if has_scanned_pages:
            self.scanned_files.append(file_path.name)
        log_event(logger, "pdf_loaded", duration_ms=timed_ms(start), data={"file_name": file_path.name, "pages": len(pages), "scanned": has_scanned_pages})
        return pages