from abc import ABC, abstractmethod

from src.models.schemas import Chunk


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int, filters: dict[str, object] | None = None) -> list[Chunk]: ...

    @abstractmethod
    def delete_by_file(self, file_name: str) -> None: ...

    @abstractmethod
    def list_files(self) -> list[dict[str, object]]: ...

    @abstractmethod
    def all_chunks(self) -> list[Chunk]: ...