from uuid import UUID

from pydantic import Field

from src.contracts.schemas import (
    ContractModel,
    QueryResult,
    SourceCitation,
)


class TokenEvent(ContractModel):
    """Internal engine event; the API adds request_id in the SSE data."""

    delta: str


class CompletedEvent(ContractModel):
    """Internal engine completion; the API serializes its result as done data."""

    result: QueryResult


class TokenSSEData(ContractModel):
    request_id: UUID
    delta: str


class CitationsEvent(ContractModel):
    request_id: UUID
    citations: list[SourceCitation] = Field(default_factory=list)


QueryStreamEvent = TokenEvent | CompletedEvent
