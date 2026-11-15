from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.models.schemas import Chunk, Source


class GraphState(TypedDict, total=False):
    question: str
    standalone_question: str
    route: str
    route_confidence: float
    route_reason: str
    chunks: list[Chunk]
    scores: list[float]
    answer: str
    sources: list[Source]
    messages: Annotated[list[BaseMessage], add_messages]
    request_id: str
    error: str
    streaming: bool
    filters: dict[str, object]
    trace: dict[str, object]