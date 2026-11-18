from uuid import UUID

import pytest
from pydantic import ValidationError

from src.contracts.errors import ErrorDetail, ErrorResponse
from src.contracts.events import CitationsEvent, CompletedEvent, TokenSSEData
from src.contracts.schemas import (
    CreateSessionResponse,
    DocumentInfo,
    EmptyRequest,
    HealthResponse,
    IngestDocumentResponse,
    IngestResult,
    ListDocumentsResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
    SourceCitation,
    UploadDocumentRequest,
)


DOCUMENT_ID = "a" * 64
REQUEST_ID = UUID("12345678-1234-5678-1234-567812345678")


def test_health_and_session_contracts() -> None:
    assert HealthResponse().status == "ok"
    assert CreateSessionResponse().session_id == "local"
    assert EmptyRequest().model_dump() == {}


def test_upload_contract_accepts_pdf_and_rejects_invalid_inputs() -> None:
    upload = UploadDocumentRequest(file_name="Report.PDF", content=b"%PDF-1.7")
    assert upload.file_name == "Report.PDF"

    with pytest.raises(ValidationError):
        UploadDocumentRequest(file_name="report.txt", content=b"text")
    with pytest.raises(ValidationError):
        UploadDocumentRequest(file_name="report.pdf", content=b"")
    with pytest.raises(ValidationError):
        UploadDocumentRequest(file_name="report.pdf", content=b"x", unexpected=True)


def test_document_and_ingest_contracts() -> None:
    document = DocumentInfo(
        document_id=DOCUMENT_ID,
        file_name="report.pdf",
        indexed_page_count=3,
        chunk_count=8,
    )
    ingest = IngestDocumentResponse(
        request_id=REQUEST_ID,
        document=document,
        duplicate=False,
    )
    assert ingest.document_id if hasattr(ingest, "document_id") else ingest.document == document

    listed = ListDocumentsResponse(request_id=REQUEST_ID, documents=[document])
    assert listed.documents[0].document_id == DOCUMENT_ID

    with pytest.raises(ValidationError):
        DocumentInfo(document_id="not-a-sha256", file_name="report.pdf", indexed_page_count=1, chunk_count=1)


def test_query_contract_normalizes_text_and_rejects_duplicate_filters() -> None:
    request = QueryRequest(query="  What changed?  ", document_ids=[DOCUMENT_ID])
    assert request.query == "What changed?"

    with pytest.raises(ValidationError):
        QueryRequest(query=" ")
    with pytest.raises(ValidationError):
        QueryRequest(query="valid", document_ids=[DOCUMENT_ID, DOCUMENT_ID])
    with pytest.raises(ValidationError):
        QueryRequest(query="valid", document_ids=["bad-id"])


def test_query_result_and_citation_contract() -> None:
    citation = SourceCitation(
        document_id=DOCUMENT_ID,
        file_name="report.pdf",
        page=2,
        snippet="The result...",
        score=0.91,
    )
    result = QueryResponse(
        request_id=REQUEST_ID,
        conversation_id=UUID("22345678-1234-5678-1234-567812345678"),
        status="answered",
        route="rag_query",
        answer="The result was positive [report.pdf, p.2].",
        citations=[citation],
    )
    assert result.citations[0].page == 2
    assert 0 <= (result.citations[0].score or 0) <= 1

    with pytest.raises(ValidationError):
        SourceCitation(document_id=DOCUMENT_ID, file_name="report.pdf", page=0, snippet="bad")


def test_error_contract_uses_shared_error_code_literal() -> None:
    error = ErrorResponse(
        request_id=REQUEST_ID,
        error=ErrorDetail(code="document_not_found", message="Document was not found."),
    )
    assert error.error.code == "document_not_found"

    with pytest.raises(ValidationError):
        ErrorDetail(code="unknown_code", message="Unknown")


def test_sse_payload_contracts() -> None:
    token = TokenSSEData(request_id=REQUEST_ID, delta="The ")
    citations = CitationsEvent(request_id=REQUEST_ID, citations=[])
    result = QueryResult(
        conversation_id=UUID("22345678-1234-5678-1234-567812345678"),
        status="answered",
        route="rag_query",
        answer="answer",
    )
    completed = CompletedEvent(result=result)

    assert token.delta == "The "
    assert citations.citations == []
    assert completed.result.status == "answered"
