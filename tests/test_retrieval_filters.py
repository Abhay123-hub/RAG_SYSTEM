from src.models.schemas import Chunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.vectorstore.base import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever


class FilteredVectorStore(VectorStore):
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.last_filters = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, query: str, k: int, filters: dict[str, object] | None = None) -> list[Chunk]:
        self.last_filters = filters
        names = filters.get("file_name") if filters else None
        if isinstance(names, dict):
            names = names.get("$in")
        candidates = self.chunks
        if names:
            candidates = [chunk for chunk in candidates if chunk.file_name in (names if isinstance(names, list) else [names])]
        return candidates[:k]

    def delete_by_file(self, file_name: str) -> None:
        self.chunks = [chunk for chunk in self.chunks if chunk.file_name != file_name]

    def list_files(self) -> list[dict[str, object]]:
        return []

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks)


def test_bm25_filter_limits_results_to_selected_documents() -> None:
    chunks = [
        Chunk(text="alpha revenue report", file_name="finance.pdf", page=1, chunk_id="1", file_hash="a"),
        Chunk(text="alpha project report", file_name="projects.pdf", page=1, chunk_id="2", file_hash="b"),
    ]

    result = BM25Retriever().retrieve(
        "alpha report",
        chunks,
        10,
        {"file_name": {"$in": ["finance.pdf"]}},
    )

    assert [chunk.file_name for chunk in result] == ["finance.pdf"]


def test_hybrid_retriever_passes_document_filters_to_vector_search() -> None:
    chunks = [
        Chunk(text="selected passage", file_name="selected.pdf", page=3, chunk_id="1", file_hash="a"),
        Chunk(text="excluded passage", file_name="excluded.pdf", page=4, chunk_id="2", file_hash="b"),
    ]
    store = FilteredVectorStore(chunks)
    retriever = HybridRetriever(store, k=10)
    filters = {"file_name": {"$in": ["selected.pdf"]}}

    result = retriever.retrieve("selected", filters)

    assert store.last_filters == filters
    assert {chunk.file_name for chunk in result} == {"selected.pdf"}
