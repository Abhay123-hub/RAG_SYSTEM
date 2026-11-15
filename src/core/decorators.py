import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger("src.core.execution")


def log_execution(function: F) -> F:
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = function(*args, **kwargs)
            logger.info("execution_done", extra={"event": "execution_done", "data": {"function": function.__qualname__}, "duration_ms": round((time.perf_counter() - start) * 1000)})
            return result
        except Exception:
            logger.exception("execution_failed", extra={"event": "execution_failed", "data": {"function": function.__qualname__}})
            raise

    return wrapper  # type: ignore[return-value]


def timed(function: F) -> F:
    return log_execution(function)


def retry(attempts: int = 3, exceptions: tuple[type[Exception], ...] = (Exception,)) -> Callable[[F], F]:
    """Retry selected transient failures with bounded exponential backoff."""

    def decorate(function: F) -> F:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(attempts):
                try:
                    return function(*args, **kwargs)
                except exceptions:
                    if attempt + 1 == attempts:
                        raise
                    time.sleep(2**attempt)
            raise RuntimeError("Retry loop ended unexpectedly")

        return wrapper  # type: ignore[return-value]

    return decorate