import json
from uuid import UUID

from src.contracts.events import CompletedEvent, TokenEvent, TokenSSEData, CitationsEvent
from src.contracts.schemas import QueryResult


REQUEST_ID = UUID("62345678-1234-5678-1234-567812345678")


def format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def test_sse_token_success_event() -> None:
    token = TokenSSEData(request_id=REQUEST_ID, delta="Hello")

    frame = format_sse("token", token.model_dump(mode="json"))

    assert frame.startswith("event: token\n")
    assert '"delta": "Hello"' in frame


def test_sse_done_event_contains_complete_query_result() -> None:
    completed = CompletedEvent(
        result=QueryResult(
            conversation_id=UUID("72345678-1234-5678-1234-567812345678"),
            status="answered",
            route="rag_query",
            answer="Complete answer",
        )
    )
    result_data = completed.result.model_dump(mode="json")
    result_data["request_id"] = str(REQUEST_ID)
    frame = format_sse("done", result_data)

    assert '"status": "answered"' in frame
    assert '"route": "rag_query"' in frame
    assert '"request_id"' in frame


def test_sse_citations_event_and_empty_citations() -> None:
    event = CitationsEvent(request_id=REQUEST_ID, citations=[]).model_dump(mode="json")
    frame = format_sse("citations", event)

    assert '"citations": []' in frame


def test_sse_error_event_contract() -> None:
    error = {
        "error": {"code": "upstream_model_error", "message": "The model provider failed."},
        "request_id": str(REQUEST_ID),
    }

    frame = format_sse("error", error)

    assert frame.startswith("event: error\n")
    assert '"code": "upstream_model_error"' in frame
