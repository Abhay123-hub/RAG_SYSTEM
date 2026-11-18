from typing import Literal
from uuid import UUID

from pydantic import Field

from src.contracts.schemas import ContractModel

ErrorCode = Literal[
    "invalid_request",
    "session_not_found",
    "document_not_found",
    "upload_too_large",
    "unsupported_media_type",
    "validation_error",
    "invalid_pdf",
    "no_extractable_text",
    "upstream_model_error",
    "service_unavailable",
    "internal_error",
]


class ErrorDetail(ContractModel):
    code: ErrorCode
    message: str = Field(min_length=1)


class ErrorResponse(ContractModel):
    error: ErrorDetail
    request_id: UUID
