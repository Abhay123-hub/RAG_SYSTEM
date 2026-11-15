import logging
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer

from config.settings import get_settings
from src.agents.router import QueryRouter
from src.core.logger import get_logger, log_event
from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_builder import CitationBuilder
from src.generation.query_rewriter import QueryRewriter
from src.guardrails.safety import SafetyChecker
from src.memory.history_manager import HistoryManager
from src.models.state import GraphState
from src.models.schemas import Chunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.relevance_grader import RelevanceGrader
from src.vectorstore.base import VectorStore

logger = get_logger(__name__)


@dataclass
class RAGNodes:
    router: QueryRouter
    rewriter: QueryRewriter
    retriever: HybridRetriever
    reranker: Reranker
    grader: RelevanceGrader
    generator: AnswerGenerator
    citation_builder: CitationBuilder
    vector_store: VectorStore
    safety: SafetyChecker
    history: HistoryManager

    def route(self, state: GraphState) -> GraphState:
        question = state["question"]
        allowed, refusal = self.safety.check(question)
        if not allowed:
            decision = {"route": "out_of_scope", "route_confidence": 1.0, "route_reason": refusal}
        else:
            messages = self.history.format(state.get("messages", []))
            files = [str(file["file_name"]) for file in self.vector_store.list_files()]
            routed = self.router.route(question, files, messages)
            decision = {"route": routed.route, "route_confidence": routed.confidence, "route_reason": routed.reason}
        return {**decision, "answer": "", "chunks": [], "scores": [], "sources": [], "trace": {}, "error": "", "standalone_question": ""}

    def document_check(self, state: GraphState) -> GraphState:
        if not self.vector_store.list_files():
            return {"answer": "Please upload and process at least one PDF before asking a document question."}
        return {}

    def rewrite(self, state: GraphState) -> GraphState:
        rewritten = self.rewriter.rewrite(state["question"], state.get("messages", []))
        return {"standalone_question": rewritten}

    def retrieve(self, state: GraphState) -> GraphState:
        return {"chunks": self.retriever.retrieve(state.get("standalone_question", state["question"]), state.get("filters") or None)}

    def rerank(self, state: GraphState) -> GraphState:
        candidates = state.get("chunks", [])
        try:
            ranked = self.reranker.rerank(state.get("standalone_question", state["question"]), candidates)
        except Exception:
            # A ranking service outage must not discard the hybrid retriever's ordering.
            logger.exception("reranker_fallback", extra={"event": "reranker_fallback"})
            ranked = candidates[: get_settings().rerank_top_n]
        return {"chunks": ranked, "scores": [chunk.score or 0.0 for chunk in ranked]}

    def grade(self, state: GraphState) -> GraphState:
        try:
            passed, score, reason = self.grader.grade(state["standalone_question"], state.get("chunks", []))
        except Exception as error:
            logger.exception("grader_failed", extra={"event": "grader_failed"})
            passed, score, reason = False, 0.0, str(error)
        trace = {**state.get("trace", {}), "grade_score": score, "grade_reason": reason}
        return {"trace": trace, "error": "" if passed else "insufficient_context"}

    def answer(self, state: GraphState) -> GraphState:
        callback = None
        if state.get("streaming"):
            writer = get_stream_writer()
            callback = lambda token: writer({"token": token})
        answer = self.generator.generate(state["standalone_question"], state.get("chunks", []), on_token=callback)
        sources, invalid = self.citation_builder.build(answer, state.get("chunks", []))
        trace = {**state.get("trace", {}), "invalid_citations": invalid}
        return {"answer": answer, "sources": sources, "trace": trace}

    def not_found(self, state: GraphState) -> GraphState:
        return {"answer": "I couldn't find enough information in the uploaded documents to answer that. Try rephrasing the question or upload a relevant PDF.", "sources": []}

    def general_chat(self, state: GraphState) -> GraphState:
        model = ChatOpenAI(model=get_settings().llm_model, temperature=0, api_key=get_settings().openai_api_key.get_secret_value())
        messages: list[BaseMessage] = self.history.trim(state.get("messages", []))
        try:
            result = model.invoke(messages or [AIMessage(content="Hello!")])
            return {"answer": str(result.content), "sources": []}
        except Exception:
            logger.exception("general_chat_failed", extra={"event": "general_chat_failed"})
            return {"answer": "I'm having trouble responding right now. Please try again shortly.", "sources": []}

    def out_of_scope(self, state: GraphState) -> GraphState:
        return {"answer": "I can't help with that request. I can help with questions about your PDFs or general topics.", "sources": []}

    def format_response(self, state: GraphState) -> GraphState:
        log_event(logger, "response_ready", data={"route": state.get("route"), "source_count": len(state.get("sources", []))})
        answer = state.get("answer", "")
        return {
            "messages": [AIMessage(content=answer)],
            "trace": {**state.get("trace", {}), "route": state.get("route"), "standalone_question": state.get("standalone_question", "")},
        }