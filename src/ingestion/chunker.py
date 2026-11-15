from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from src.models.schemas import Chunk, PageContent


class DocumentChunker:
    """Split each page independently so every chunk retains an exact citation page."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        settings = get_settings()
        size = chunk_size or settings.chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)

    def chunk(self, pages: list[PageContent]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            texts = self.splitter.split_text(page.text)
            for index, text in enumerate(texts):
                chunks.append(
                    Chunk(
                        text=text,
                        file_name=page.file_name,
                        page=page.page,
                        chunk_id=f"{page.file_hash}:{page.page}:{index}",
                        file_hash=page.file_hash,
                    )
                )
        return chunks