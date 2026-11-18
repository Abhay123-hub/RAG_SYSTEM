from uuid import UUID

from src.contracts.events import CompletedEvent, TokenEvent
from src.contracts.schemas import (
    DocumentInfo,
    IngestDocumentResponse,
    IngestResult,
    ListDocumentsResponse,
    QueryResponse,
)


class MockRAGEngine:
    def __init__(self) -> None:
        self.document = DocumentInfo(
            document_id="b" * 64,
            file_name="integration.pdf",
            indexed_page_count=2,
            chunk_count=4,
        )
        self.deleted = []

    def initialize_session(self, *, session_id: str) -> None:
        assert session_id == "local"

    def list_documents(self, *, session_id: str) -> list[DocumentInfo]:
        return [self.document]

    def ingest_document(self, *, session_id: str, file_name: str, content: bytes) -> IngestResult:
        return IngestResult(document=self.document, duplicate=False)

    def delete_document(self, *, session_id: str, document_id: str) -> None:
        self.deleted.append(document_id)

    def query(self, **kwargs) -> dict:
        return {
            "conversation_id": kwargs["conversation_id"],
            "status": "answered",
            "route": "rag_query",
            "answer": "Answer [integration.pdf, p.1]",
            "citations": [],
            "trace": {"retrieval": "mocked"},
        }

    def stream_query(self, **kwargs):
        yield TokenEvent(delta="Answer ")
        yield CompletedEvent(result=kwargs["result"])


def test_ingest_list_and_delete_integration_flow() -> None:
    engine = MockRAGEngine()
    request_id = UUID("32345678-1234-5678-1234-567812345678")
    engine.initialize_session(session_id="local")

    ingest = IngestDocumentResponse(
        request_id=request_id,
        document=engine.document,
        duplicate=False,
    )
    assert ingest.document.file_name == "integration.pdf"

    listed = ListDocumentsResponse(request_id=request_id, documents=engine.list_documents(session_id="local"))
    assert len(listed.documents) == 1

    engine.delete_document(session_id="local", document_id=engine.document.document_id)
    assert engine.deleted == ["b" * 64]


def test_query_integration_flow_with_mock_engine() -> None:
    engine = MockRAGEngine()
    conversation_id = UUID("42345678-1234-5678-1234-567812345678")

    result = engine.query(
        session_id="local",
        conversation_id=conversation_id,
        request_id=UUID("52345678-1234-5678-1234-567812345678"),
        query="What is the integration result?",
        document_ids=["b" * 64],
        include_trace=True,
    )

    response = QueryResponse(request_id=UUID("52345678-1234-5678-1234-567812345678"), **result)
    assert response.status == "answered"
    assert response.route == "rag_query"
    assert response.trace["retrieval"] == "mocked"
