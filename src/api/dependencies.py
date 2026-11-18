from fastapi import Request

from src.contracts.engine import RAGEngine


def get_engine(request: Request) -> RAGEngine:
    return request.app.state.rag_engine
