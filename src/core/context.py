from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator


request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
session_id_var: ContextVar[str] = ContextVar("session_id", default="-")


@contextmanager
def request_context(request_id: str, session_id: str) -> Iterator[None]:
    request_token: Token[str] = request_id_var.set(request_id)
    session_token: Token[str] = session_id_var.set(session_id)
    try:
        yield
    finally:
        request_id_var.reset(request_token)
        session_id_var.reset(session_token)