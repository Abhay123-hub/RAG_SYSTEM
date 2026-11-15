from src.models.schemas import Chunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vectorstore.base import VectorStore


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(text=text, file_name="guide.pdf", page=1, chunk_id=chunk_id, file_hash="hash")


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.chunks = [make_chunk("a", "hybrid search combines keyword and vector results"), make_chunk("b", "another passage about vector retrieval")]

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, query: str, k: int, filters=None) -> list[Chunk]:
        return [self.chunks[0], self.chunks[1]][:k]

    def delete_by_file(self, file_name: str) -> None:
        self.chunks = [chunk for chunk in self.chunks if chunk.file_name != file_name]

    def list_files(self) -> list[dict[str, object]]:
        return []

    def all_chunks(self) -> list[Chunk]:
        return self.chunks


def test_bm25_retrieves_exact_terms() -> None:
    chunks = [make_chunk("a", "alpha exact terminology"), make_chunk("b", "unrelated content"), make_chunk("c", "third distinct passage")]

    result = BM25Retriever().retrieve("alpha terminology", chunks, 1)

    assert result[0].chunk_id == "a"


def test_hybrid_retriever_merges_vector_and_keyword_results() -> None:
    result = HybridRetriever(FakeVectorStore(), k=2).retrieve("hybrid keyword search")

    assert result[0].chunk_id == "a"
    assert len(result) == 2
    assert result[0].score is not None