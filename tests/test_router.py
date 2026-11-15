from langchain_core.runnables import RunnableLambda

from src.agents.router import QueryRouter, RouteDecision


def test_low_confidence_router_defaults_to_rag() -> None:
    model = RunnableLambda(lambda _: RouteDecision(route="general_chat", confidence=0.2, reason="uncertain"))

    decision = QueryRouter(model=model).route("What does the report say?", ["report.pdf"])

    assert decision.route == "rag_query"