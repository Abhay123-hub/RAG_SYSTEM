import re

from rank_bm25 import BM25Okapi

from src.models.schemas import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class BM25Retriever:
    """Keyword retrieval over the active session's indexed chunks."""

    def retrieve(self, query: str, chunks: list[Chunk], k: int, filters: dict[str, object] | None = None) -> list[Chunk]:
        if filters and "file_hash" in filters:
            allowed = filters["file_hash"]
            if isinstance(allowed, dict):
                allowed = allowed.get("$in", [])
            hashes = set(allowed if isinstance(allowed, list) else [allowed])
            chunks = [chunk for chunk in chunks if chunk.file_hash in hashes]
        if filters and "file_name" in filters:
            allowed = filters["file_name"]
            if isinstance(allowed, dict):
                allowed = allowed.get("$in", [])
            names = set(allowed if isinstance(allowed, list) else [allowed])
            chunks = [chunk for chunk in chunks if chunk.file_name in names]
        if not chunks:
            return []
        scores = BM25Okapi([tokenize(chunk.text) for chunk in chunks]).get_scores(tokenize(query))
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda item: float(item[1]), reverse=True)
        return [chunk.model_copy(update={"score": float(score)}) for chunk, score in ranked[:k]]