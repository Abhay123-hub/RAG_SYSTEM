import time

from config.settings import get_settings
from src.core.exceptions import RetrievalError
from src.core.logger import get_logger, log_event, timed_ms
from src.models.schemas import Chunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.vectorstore.base import VectorStore

logger = get_logger(__name__)


class HybridRetriever:
    """Merge semantic and keyword rankings with reciprocal-rank fusion."""

    def __init__(self, vector_store: VectorStore, bm25: BM25Retriever | None = None, k: int | None = None) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25 or BM25Retriever()
        self.k = k or get_settings().retrieval_k

    def retrieve(self, query: str, filters: dict[str, object] | None = None) -> list[Chunk]:
        start = time.perf_counter()
        try:
            vector_hits = self.vector_store.search(query, self.k, filters)
            keyword_hits = self.bm25.retrieve(query, self.vector_store.all_chunks(), self.k, filters)
        except Exception as error:
            raise RetrievalError(f"Hybrid retrieval failed: {error}") from error

        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for ranking in (vector_hits, keyword_hits):
            for rank, chunk in enumerate(ranking, start=1):
                # Reciprocal-rank fusion avoids mixing incomparable vector and BM25 scores.
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (60 + rank)
                chunks[chunk.chunk_id] = chunk
        merged = [chunks[key].model_copy(update={"score": score}) for key, score in scores.items()]
        merged.sort(key=lambda chunk: chunk.score or 0, reverse=True)
        log_event(logger, "retrieval_done", duration_ms=timed_ms(start), data={"vector_hits": len(vector_hits), "bm25_hits": len(keyword_hits), "merged": len(merged), "top_score": merged[0].score if merged else None})
        return merged[: self.k]