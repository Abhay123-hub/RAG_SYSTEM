import pymupdf

from src.ingestion.file_registry import FileRegistry
from src.ingestion.ingestion_service import IngestionService
from src.ingestion.chunker import DocumentChunker
from src.ingestion.pdf_loader import PDFLoader
from src.models.schemas import Chunk
from src.vectorstore.base import VectorStore


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, query: str, k: int, filters: dict[str, object] | None = None) -> list[Chunk]:
        return self.chunks[:k]

    def delete_by_file(self, file_name: str) -> None:
        self.chunks = [chunk for chunk in self.chunks if chunk.file_name != file_name]

    def list_files(self) -> list[dict[str, object]]:
        entries: dict[str, dict[str, object]] = {}
        for chunk in self.chunks:
            entry = entries.setdefault(
                chunk.file_hash,
                {"file_hash": chunk.file_hash, "file_name": chunk.file_name, "pages": set(), "chunks": 0},
            )
            entry["pages"].add(chunk.page)
            entry["chunks"] = int(entry["chunks"]) + 1
        return [
            {
                "file_hash": key,
                "file_name": entry["file_name"],
                "pages": len(entry["pages"]),
                "chunks": entry["chunks"],
            }
            for key, entry in entries.items()
        ]

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks)


def make_pdf(path, text: str) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def test_pdf_upload_validation_accepts_extractable_pdf(tmp_path) -> None:
    pdf_path = tmp_path / "valid.pdf"
    make_pdf(pdf_path, "This PDF contains extractable text.")

    pages = PDFLoader().load(pdf_path)

    assert len(pages) == 1
    assert "extractable text" in pages[0].text


def test_pdf_upload_validation_rejects_invalid_or_empty_pdf(tmp_path) -> None:
    invalid_path = tmp_path / "invalid.pdf"
    invalid_path.write_bytes(b"not a pdf")

    try:
        PDFLoader().load(invalid_path)
    except Exception as error:
        assert "Could not read" in str(error)
    else:
        raise AssertionError("Invalid PDF bytes must fail validation")

    empty_path = tmp_path / "empty.pdf"
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(empty_path)
    pdf.close()
    pages = PDFLoader().load(empty_path)
    assert pages == []


def test_duplicate_filename_with_different_content_is_indexed_separately(tmp_path) -> None:
    store = InMemoryVectorStore()
    service = IngestionService(PDFLoader(), DocumentChunker(chunk_size=100, chunk_overlap=10), FileRegistry(), store)
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "same-name.pdf"
    second = second_dir / "same-name.pdf"
    make_pdf(first, "The first document describes apples.")
    make_pdf(second, "The second document describes oranges.")

    first_result = service.ingest(first)
    second_result = service.ingest(second)

    assert first_result["duplicate"] is False
    assert second_result["duplicate"] is False
    assert first_result["file_hash"] != second_result["file_hash"]
    assert len(store.list_files()) == 2
    assert {entry["file_name"] for entry in store.list_files()} == {"same-name.pdf"}


def test_identical_duplicate_content_is_reported_without_reindexing(tmp_path) -> None:
    store = InMemoryVectorStore()
    service = IngestionService(PDFLoader(), DocumentChunker(chunk_size=100, chunk_overlap=10), FileRegistry(), store)
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    make_pdf(first, "Identical document content.")
    second.write_bytes(first.read_bytes())

    initial = service.ingest(first)
    duplicate = service.ingest(second)

    assert initial["duplicate"] is False
    assert duplicate["duplicate"] is True
    assert len(store.list_files()) == 1
    assert len(store.chunks) == 1


def test_document_listing_and_deletion() -> None:
    store = InMemoryVectorStore()
    store.add_chunks([
        Chunk(text="A", file_name="a.pdf", page=1, chunk_id="a", file_hash="1" * 64),
        Chunk(text="B", file_name="b.pdf", page=1, chunk_id="b", file_hash="2" * 64),
    ])

    listed = store.list_files()
    assert {entry["file_name"] for entry in listed} == {"a.pdf", "b.pdf"}

    store.delete_by_file("a.pdf")
    assert [chunk.file_name for chunk in store.all_chunks()] == ["b.pdf"]
