import time

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.nodes import RAGNodes
from src.models.state import GraphState


class RAGGraphBuilder:
    """Assemble the router, RAG, fallback, and response-formatting graph."""

    def __init__(self, nodes: RAGNodes, checkpointer: MemorySaver | None = None) -> None:
        self.nodes = nodes
        self.checkpointer = checkpointer or MemorySaver()

    def build(self):
        graph = StateGraph(GraphState)

        def timed(name: str, node):
            def run(state: GraphState) -> GraphState:
                start = time.perf_counter()
                result = node(state)
                timings = dict(state.get("trace", {}).get("timings_ms", {}))
                timings[name] = round((time.perf_counter() - start) * 1000)
                trace = {**state.get("trace", {}), **result.get("trace", {}), "timings_ms": timings}
                return {**result, "trace": trace}

            return run

        graph.add_node("router", timed("router", self.nodes.route))
        graph.add_node("document_check", timed("document_check", self.nodes.document_check))
        graph.add_node("query_rewriter", timed("query_rewriter", self.nodes.rewrite))
        graph.add_node("hybrid_retriever", timed("hybrid_retriever", self.nodes.retrieve))
        graph.add_node("reranker", timed("reranker", self.nodes.rerank))
        graph.add_node("relevance_grader", timed("relevance_grader", self.nodes.grade))
        graph.add_node("answer_generator", timed("answer_generator", self.nodes.answer))
        graph.add_node("not_found", timed("not_found", self.nodes.not_found))
        graph.add_node("general_chat", timed("general_chat", self.nodes.general_chat))
        graph.add_node("out_of_scope", timed("out_of_scope", self.nodes.out_of_scope))
        graph.add_node("response_formatter", timed("response_formatter", self.nodes.format_response))

        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            lambda state: state["route"],
            {"rag_query": "document_check", "general_chat": "general_chat", "out_of_scope": "out_of_scope"},
        )
        graph.add_conditional_edges(
            "document_check",
            lambda state: "ready" if (
                self.nodes.vector_store.list_files()
                and not state.get("filters")
                or state.get("filters")
                and any(
                    chunk.file_hash in (
                        state["filters"]["file_hash"].get("$in", [])
                        if isinstance(state["filters"].get("file_hash"), dict)
                        else [state["filters"].get("file_hash")]
                    )
                    for chunk in self.nodes.vector_store.all_chunks()
                )
            ) else "empty",
            {"ready": "query_rewriter", "empty": "response_formatter"},
        )
        graph.add_edge("query_rewriter", "hybrid_retriever")
        graph.add_edge("hybrid_retriever", "reranker")
        graph.add_edge("reranker", "relevance_grader")
        graph.add_conditional_edges(
            "relevance_grader",
            lambda state: "answer" if not state.get("error") else "not_found",
            {"answer": "answer_generator", "not_found": "not_found"},
        )
        for node in ("answer_generator", "not_found", "general_chat", "out_of_scope"):
            graph.add_edge(node, "response_formatter")
        graph.add_edge("response_formatter", END)
        return graph.compile(checkpointer=self.checkpointer)