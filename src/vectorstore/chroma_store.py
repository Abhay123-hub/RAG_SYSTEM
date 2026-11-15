import hashlib
from collections import defaultdict
from pathlib import Path

from langchain_chroma import Chroma

from config.settings import get_settings
from src.models.schemas import Chunk
from src.vectorstore.base import VectorStore


class ChromaVectorStore(VectorStore):
    """Persistent Chroma collection isolated to one chat session."""

    def __init__(self, session_id: str, embeddings: object) -> None:
        settings = get_settings()
        collection_id = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:20]
        Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
        self.store = Chroma(
            collection_name=f"rag_{collection_id}",
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_dir),
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if chunks:
            self.store.add_documents(
                [chunk.to_document() for chunk in chunks],
                ids=[chunk.chunk_id for chunk in chunks],
            )

    def search(self, query: str, k: int, filters: dict[str, object] | None = None) -> list[Chunk]:
        matches = self.store.similarity_search_with_relevance_scores(query, k=k, filter=filters)
        return [Chunk.from_document(document, score=float(score)) for document, score in matches]

    def delete_by_file(self, file_name: str) -> None:
        self.store._collection.delete(where={"file_name": file_name})

    def list_files(self) -> list[dict[str, object]]:
        records = self.store._collection.get(include=["metadatas"])
        counts: dict[str, dict[str, object]] = defaultdict(dict)
        for metadata in records.get("metadatas") or []:
            if metadata is None:
                continue
            file_hash = str(metadata["file_hash"])
            entry = counts.setdefault(file_hash, {"file_hash": file_hash, "file_name": metadata["file_name"], "pages": set(), "chunks": 0})
            entry["pages"].add(int(metadata["page"]))
            entry["chunks"] += 1
        return [
            {"file_hash": key, "file_name": value["file_name"], "pages": len(value["pages"]), "chunks": value["chunks"]}
            for key, value in counts.items()
        ]

    def all_chunks(self) -> list[Chunk]:
        records = self.store._collection.get(include=["documents", "metadatas"])
        return [
            Chunk(
                text=text,
                file_name=str(metadata["file_name"]),
                page=int(metadata["page"]),
                chunk_id=str(metadata["chunk_id"]),
                file_hash=str(metadata["file_hash"]),
            )
            for text, metadata in zip(records.get("documents") or [], records.get("metadatas") or [], strict=True)
            if text is not None and metadata is not None
        ]