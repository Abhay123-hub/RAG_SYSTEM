from collections.abc import Iterator, Sequence
from typing import Protocol
from uuid import UUID

from src.contracts.events import QueryStreamEvent
from src.contracts.schemas import DocumentInfo, IngestResult, QueryResult


class RAGEngine(Protocol):
    """Public boundary between the API service and the RAG implementation."""

    def initialize_session(self, *, session_id: str) -> None: ...

    def list_documents(self, *, session_id: str) -> list[DocumentInfo]: ...

    def ingest_document(
        self, *, session_id: str, file_name: str, content: bytes
    ) -> IngestResult: ...

    def delete_document(self, *, session_id: str, document_id: str) -> None: ...

    def query(
        self,
        *,
        session_id: str,
        conversation_id: UUID,
        request_id: UUID,
        query: str,
        document_ids: Sequence[str] | None,
        include_trace: bool,
    ) -> QueryResult: ...

    def stream_query(
        self,
        *,
        session_id: str,
        conversation_id: UUID,
        request_id: UUID,
        query: str,
        document_ids: Sequence[str] | None,
        include_trace: bool,
    ) -> Iterator[QueryStreamEvent]: ...
