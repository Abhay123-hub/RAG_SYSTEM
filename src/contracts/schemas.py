from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

MAX_UPLOAD_BYTES = 50 * 1024 * 1024

DocumentId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
QueryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=10_000),
]
DocumentIdList = Annotated[list[DocumentId], Field(min_length=1)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyRequest(ContractModel):
    """Request body for endpoints that accept no fields."""


class HealthResponse(ContractModel):
    status: Literal["ok"] = "ok"


class CreateSessionResponse(ContractModel):
    session_id: Literal["local"] = "local"


class DocumentInfo(ContractModel):
    document_id: DocumentId
    file_name: str = Field(min_length=1)
    indexed_page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)


class IngestResult(ContractModel):
    document: DocumentInfo
    duplicate: bool


class UploadDocumentRequest(ContractModel):
    """Validated upload payload after the multipart file has been read.

    The HTTP request contains one multipart field named ``file``. Its
    multipart filename and bytes are normalized to these fields by the API.
    """

    file_name: str = Field(min_length=1, max_length=255)
    content: bytes = Field(min_length=1, max_length=MAX_UPLOAD_BYTES)

    @field_validator("file_name")
    @classmethod
    def require_pdf_filename(cls, value: str) -> str:
        if not value.casefold().endswith(".pdf"):
            raise ValueError("The uploaded filename must end in .pdf.")
        return value


class ListDocumentsResponse(ContractModel):
    request_id: UUID
    documents: list[DocumentInfo]


class IngestDocumentResponse(IngestResult):
    request_id: UUID


class QueryRequest(ContractModel):
    query: QueryText
    conversation_id: UUID | None = None
    document_ids: DocumentIdList | None = None
    include_trace: bool = False

    @field_validator("document_ids")
    @classmethod
    def require_unique_document_ids(
        cls, value: list[str] | None
    ) -> list[str] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("document_ids must not contain duplicates.")
        return value


class SourceCitation(ContractModel):
    document_id: DocumentId
    file_name: str = Field(min_length=1)
    page: int = Field(ge=1)
    snippet: str
    score: float | None = Field(default=None, ge=0, le=1)


class QueryResult(ContractModel):
    conversation_id: UUID
    status: Literal["answered", "insufficient_context", "refused"]
    route: Literal["rag_query", "general_chat", "out_of_scope"]
    answer: str
    citations: list[SourceCitation] = Field(default_factory=list)
    trace: dict[str, object] | None = None


class QueryResponse(QueryResult):
    request_id: UUID
