import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.contracts.errors import ErrorResponse
from src.services.errors import APIError

logger = logging.getLogger(__name__)

STATUS_BY_CODE = {
    "invalid_request": 400,
    "session_not_found": 404,
    "document_not_found": 404,
    "upload_too_large": 413,
    "unsupported_media_type": 415,
    "validation_error": 422,
    "invalid_pdf": 422,
    "no_extractable_text": 422,
    "upstream_model_error": 502,
    "service_unavailable": 503,
    "internal_error": 500,
}


def error_response(request: Request, code: str, message: str, status: int) -> JSONResponse:
    body = ErrorResponse(
        error={"code": code, "message": message},
        request_id=request.state.request_id,
    )
    return JSONResponse(
        status_code=status,
        content=body.model_dump(mode="json"),
        headers={"X-Request-ID": str(request.state.request_id)},
    )


async def handle_api_error(request: Request, error: APIError) -> JSONResponse:
    return error_response(
        request, error.code, error.message, STATUS_BY_CODE[error.code]
    )


async def handle_validation_error(
    request: Request, _error: RequestValidationError
) -> JSONResponse:
    return error_response(
        request,
        "validation_error",
        "The request does not match the API contract.",
        STATUS_BY_CODE["validation_error"],
    )


async def handle_http_error(
    request: Request, error: StarletteHTTPException
) -> JSONResponse:
    code = "invalid_request" if error.status_code < 500 else "internal_error"
    message = (
        str(error.detail)
        if isinstance(error.detail, str)
        else "The request could not be completed."
    )
    return error_response(request, code, message, error.status_code)


async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    logger.exception(
        "api_request_failed",
        extra={"event": "api_request_failed", "data": {"request_id": str(request.state.request_id)}},
        exc_info=error,
    )
    return error_response(
        request,
        "internal_error",
        "The request could not be completed.",
        STATUS_BY_CODE["internal_error"],
    )
