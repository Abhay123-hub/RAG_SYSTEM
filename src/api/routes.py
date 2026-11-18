import json
import logging
import re
import uuid
from pathlib import PurePosixPath
from typing import Annotated, Iterator
from uuid import UUID

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ValidationError

from src.api.dependencies import get_engine
from src.contracts.engine import RAGEngine
from src.contracts.errors import ErrorResponse
from src.contracts.events import (
    CitationsEvent,
    CompletedEvent,
    TokenEvent,
    TokenSSEData,
)
from src.contracts.schemas import (
    CreateSessionResponse,
    EmptyRequest,
    HealthResponse,
    IngestDocumentResponse,
    ListDocumentsResponse,
    MAX_UPLOAD_BYTES,
    QueryRequest,
    QueryResponse,
    UploadDocumentRequest,
)
from src.services.errors import APIError

logger = logging.getLogger(__name__)
router = APIRouter()
SESSION_ID = "local"
_DOCUMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _session(session_id: str) -> str:
    if session_id != SESSION_ID:
        raise APIError("session_not_found", "The requested session was not found.")
    return session_id


def _query_request(
    body: QueryRequest, request_id: UUID, engine: RAGEngine
) -> QueryResponse:
    result = engine.query(
        session_id=SESSION_ID,
        conversation_id=body.conversation_id or uuid.uuid4(),
        request_id=request_id,
        query=body.query,
        document_ids=body.document_ids,
        include_trace=body.include_trace,
    )
    return QueryResponse(request_id=request_id, **result.model_dump())


def _validate_document_ids(
    session_id: str, document_ids: list[str] | None, engine: RAGEngine
) -> None:
    if document_ids is None:
        return
    try:
        available_ids = {
            document.document_id
            for document in engine.list_documents(session_id=session_id)
        }
    except KeyError as error:
        raise APIError(
            "session_not_found", "The requested session was not found."
        ) from error
    if not set(document_ids).issubset(available_ids):
        raise APIError(
            "document_not_found",
            "One or more requested documents were not found.",
        )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.post(
    "/api/sessions",
    response_model=CreateSessionResponse,
    status_code=201,
)
def create_session(
    _body: EmptyRequest | None = None,
    engine: RAGEngine = Depends(get_engine),
) -> CreateSessionResponse:
    try:
        engine.initialize_session(session_id=SESSION_ID)
    except Exception as error:
        logger.exception("session_initialization_failed", extra={"event": "session_initialization_failed"})
        raise APIError(
            "service_unavailable", "The session could not be initialized."
        ) from error
    return CreateSessionResponse()


@router.get(
    "/api/sessions/{session_id}/documents",
    response_model=ListDocumentsResponse,
)
def list_documents(
    session_id: str,
    request: Request,
    engine: RAGEngine = Depends(get_engine),
) -> ListDocumentsResponse:
    _session(session_id)
    try:
        documents = engine.list_documents(session_id=session_id)
    except KeyError as error:
        raise APIError("session_not_found", "The requested session was not found.") from error
    return ListDocumentsResponse(request_id=request.state.request_id, documents=documents)


@router.post(
    "/api/sessions/{session_id}/documents",
    response_model=IngestDocumentResponse,
    status_code=201,
)
async def upload_document(
    session_id: str,
    request: Request,
    file: Annotated[UploadFile, File()],
    engine: RAGEngine = Depends(get_engine),
) -> IngestDocumentResponse:
    _session(session_id)
    content_type = (file.content_type or "").casefold()
    if content_type not in {"application/pdf", "application/octet-stream"}:
        raise APIError(
            "unsupported_media_type", "The uploaded file must be a PDF."
        )
    raw_name = (file.filename or "").replace("\\", "/")
    file_name = PurePosixPath(raw_name).name
    try:
        request_body = UploadDocumentRequest(
            file_name=file_name,
            content=await file.read(MAX_UPLOAD_BYTES + 1),
        )
    except ValidationError as error:
        code = (
            "upload_too_large"
            if len(error.errors()) == 1
            and error.errors()[0]["loc"] == ("content",)
            and error.errors()[0]["type"] == "bytes_too_long"
            else "validation_error"
        )
        message = (
            "The uploaded file exceeds the maximum allowed size."
            if code == "upload_too_large"
            else "The uploaded file must have a valid PDF filename and non-empty content."
        )
        raise APIError(code, message) from error
    if len(request_body.content) > MAX_UPLOAD_BYTES:
        raise APIError(
            "upload_too_large", "The uploaded file exceeds the maximum allowed size."
        )

    try:
        result = await run_in_threadpool(
            engine.ingest_document,
            session_id=session_id,
            file_name=request_body.file_name,
            content=request_body.content,
        )
    except KeyError as error:
        raise APIError("session_not_found", "The requested session was not found.") from error
    except ValueError as error:
        raise APIError("no_extractable_text", str(error)) from error
    except RuntimeError as error:
        raise APIError("invalid_pdf", str(error)) from error
    except Exception as error:
        logger.exception("document_ingestion_failed", extra={"event": "document_ingestion_failed"})
        raise APIError(
            "service_unavailable", "The document could not be indexed."
        ) from error
    return IngestDocumentResponse(
        request_id=request.state.request_id,
        document=result.document,
        duplicate=result.duplicate,
    )


@router.delete(
    "/api/sessions/{session_id}/documents/{document_id}",
    status_code=204,
    response_class=Response,
)
def delete_document(
    session_id: str,
    document_id: str,
    engine: RAGEngine = Depends(get_engine),
) -> Response:
    _session(session_id)
    if not _DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise APIError("invalid_request", "The document_id is not valid.")
    try:
        documents = engine.list_documents(session_id=session_id)
        if not any(item.document_id == document_id for item in documents):
            raise APIError("document_not_found", "The requested document was not found.")
        engine.delete_document(session_id=session_id, document_id=document_id)
    except KeyError as error:
        raise APIError("session_not_found", "The requested session was not found.") from error
    return Response(status_code=204)


@router.post(
    "/api/sessions/{session_id}/query",
    response_model=QueryResponse,
)
def query(
    session_id: str,
    body: QueryRequest,
    request: Request,
    engine: RAGEngine = Depends(get_engine),
) -> QueryResponse:
    _session(session_id)
    _validate_document_ids(session_id, body.document_ids, engine)
    try:
        return _query_request(body, request.state.request_id, engine)
    except KeyError as error:
        raise APIError("session_not_found", "The requested session was not found.") from error
    except Exception as error:
        logger.exception("query_failed", extra={"event": "query_failed"})
        raise APIError("upstream_model_error", "The query could not be completed.") from error


@router.post(
    "/api/sessions/{session_id}/query/stream",
    response_class=StreamingResponse,
)
def stream_query(
    session_id: str,
    body: QueryRequest,
    request: Request,
    engine: RAGEngine = Depends(get_engine),
) -> StreamingResponse:
    _session(session_id)
    _validate_document_ids(session_id, body.document_ids, engine)
    request_id: UUID = request.state.request_id
    conversation_id = body.conversation_id or uuid.uuid4()

    def encode(event_name: str, payload: BaseModel) -> str:
        data = payload.model_dump(mode="json")
        return f"event: {event_name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"

    def events() -> Iterator[str]:
        try:
            for event in engine.stream_query(
                session_id=session_id,
                conversation_id=conversation_id,
                request_id=request_id,
                query=body.query,
                document_ids=body.document_ids,
                include_trace=body.include_trace,
            ):
                if isinstance(event, TokenEvent):
                    yield encode(
                        "token",
                        TokenSSEData(request_id=request_id, delta=event.delta),
                    )
                elif isinstance(event, CompletedEvent):
                    yield encode(
                        "citations",
                        CitationsEvent(
                            request_id=request_id,
                            citations=event.result.citations,
                        ),
                    )
                    yield encode(
                        "done",
                        QueryResponse(
                            request_id=request_id,
                            **event.result.model_dump(),
                        ),
                    )
        except Exception as error:
            logger.exception(
                "query_stream_failed", extra={"event": "query_stream_failed"}
            )
            api_error = (
                APIError("session_not_found", "The requested session was not found.")
                if isinstance(error, KeyError)
                else APIError("upstream_model_error", "The query could not be completed.")
            )
            error_body = ErrorResponse(
                error={"code": api_error.code, "message": api_error.message},
                request_id=request_id,
            )
            yield encode("error", error_body)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": str(request_id),
        },
    )
