from src.ingestion.chunker import DocumentChunker
from src.models.schemas import PageContent


def test_chunker_preserves_page_and_file_metadata() -> None:
    page = PageContent(file_name="guide.pdf", page=2, text="A useful sentence. " * 20, file_hash="hash")

    chunks = DocumentChunker(chunk_size=40, chunk_overlap=5).chunk([page])

    assert len(chunks) > 1
    assert all(chunk.page == 2 for chunk in chunks)
    assert all(chunk.file_name == "guide.pdf" and chunk.file_hash == "hash" for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)