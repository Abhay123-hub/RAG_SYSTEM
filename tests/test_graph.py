from langchain_core.messages import HumanMessage

from src.agents.graph_builder import RAGGraphBuilder
from src.agents.nodes import RAGNodes
from src.agents.router import RouteDecision
from src.generation.citation_builder import CitationBuilder
from src.guardrails.safety import SafetyChecker
from src.memory.history_manager import HistoryManager


class EmptyStore:
    def list_files(self):
        return []


class GeneralRouter:
    def route(self, question, files, history=""):
        return RouteDecision(route="general_chat", confidence=0.9, reason="Greeting")


def test_graph_routes_general_chat_without_retrieval() -> None:
    nodes = RAGNodes(
        router=GeneralRouter(),
        rewriter=object(),
        retriever=object(),
        reranker=object(),
        grader=object(),
        generator=object(),
        citation_builder=CitationBuilder(),
        vector_store=EmptyStore(),
        safety=SafetyChecker(),
        history=HistoryManager(5),
    )
    nodes.general_chat = lambda state: {"answer": "Hello there.", "sources": []}
    graph = RAGGraphBuilder(nodes).build()

    result = graph.invoke(
        {"question": "Hello", "messages": [HumanMessage(content="Hello")], "streaming": False, "filters": {}},
        {"configurable": {"thread_id": "test"}},
    )

    assert result["route"] == "general_chat"
    assert result["answer"] == "Hello there."
    assert result["trace"]["timings_ms"]["router"] >= 0