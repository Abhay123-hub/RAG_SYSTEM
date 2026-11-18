import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.errors import (
    handle_api_error,
    handle_http_error,
    handle_unexpected_error,
    handle_validation_error,
)
from src.api.routes import router
from src.contracts.engine import RAGEngine
from src.core.context import request_context
from src.services.errors import APIError
from src.services.rag_engine import RAGEngineService


def create_app(engine: RAGEngine | None = None) -> FastAPI:
    app = FastAPI(title="RAG Platform API", version="1.0.0")
    app.state.rag_engine = engine if engine is not None else RAGEngineService()
    app.include_router(router)
    app.add_exception_handler(APIError, handle_api_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_error)
    app.add_exception_handler(Exception, handle_unexpected_error)

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = uuid.uuid4()
        request.state.request_id = request_id
        request.state.session_id = "local"
        with request_context(str(request_id), "local"):
            response = await call_next(request)
        response.headers["X-Request-ID"] = str(request_id)
        return response

    return app


app = create_app()
