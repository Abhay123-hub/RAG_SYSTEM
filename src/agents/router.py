from typing import Literal

from openai import RateLimitError
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config.settings import get_settings
from src.core.decorators import retry
from src.core.llm import create_chat_model
from src.core.logger import get_logger, log_event
from src.generation.prompts import ROUTER_SYSTEM_PROMPT

logger = get_logger(__name__)


class RouteDecision(BaseModel):
    route: Literal["rag_query", "general_chat", "out_of_scope"]
    confidence: float = Field(ge=0, le=1)
    reason: str


class QueryRouter:
    """Classify messages, defaulting to document retrieval if uncertain."""

    def __init__(self, model: object | None = None) -> None:
        settings = get_settings()
        self.model = model or create_chat_model(settings.router_model).with_structured_output(RouteDecision)
        self.prompt = ChatPromptTemplate.from_messages([("system", ROUTER_SYSTEM_PROMPT), ("human", "Documents: {files}\nHistory:\n{history}\n\nLatest message: {question}")])

    @retry(attempts=3, exceptions=(RateLimitError,))
    def route(self, question: str, files: list[str], history: str = "") -> RouteDecision:
        try:
            decision: RouteDecision = (self.prompt | self.model).invoke({"question": question, "files": ", ".join(files) or "none", "history": history})
        except Exception:
            logger.exception("router_failed", extra={"event": "router_failed"})
            decision = RouteDecision(route="rag_query", confidence=0, reason="Router failed; defaulting to document search.")
        if decision.confidence < get_settings().router_confidence_threshold:
            decision = decision.model_copy(update={"route": "rag_query"})
        log_event(logger, "route_selected", data={"question": question, "route": decision.route, "confidence": decision.confidence, "reason": decision.reason})
        return decision