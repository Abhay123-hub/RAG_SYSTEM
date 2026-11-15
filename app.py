import uuid
from pathlib import Path

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver

from config.settings import get_settings
from src.agents.graph_builder import RAGGraphBuilder
from src.agents.nodes import RAGNodes
from src.agents.router import QueryRouter
from src.core.context import request_context
from src.core.logger import configure_logging
from src.embeddings.embedder import EmbeddingService
from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_builder import CitationBuilder
from src.generation.query_rewriter import QueryRewriter
from src.ingestion.chunker import DocumentChunker
from src.ingestion.file_registry import FileRegistry
from src.ingestion.ingestion_service import IngestionService
from src.ingestion.pdf_loader import PDFLoader
from src.guardrails.safety import SafetyChecker
from src.memory.history_manager import HistoryManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.relevance_grader import RelevanceGrader
from src.ui.chat_view import render_chat
from src.ui.sidebar import render_sidebar
from src.vectorstore.chroma_store import ChromaVectorStore


@st.cache_resource(show_spinner="Connecting to the document store...")
def get_runtime(session_id: str):
    settings = get_settings()
    embeddings = EmbeddingService()
    store = ChromaVectorStore(session_id, embeddings)
    registry = FileRegistry()
    ingestion = IngestionService(PDFLoader(), DocumentChunker(), registry, store)
    history = HistoryManager(settings.history_turns)
    nodes = RAGNodes(
        router=QueryRouter(),
        rewriter=QueryRewriter(history_manager=history),
        retriever=HybridRetriever(store),
        reranker=Reranker(),
        grader=RelevanceGrader(),
        generator=AnswerGenerator(),
        citation_builder=CitationBuilder(),
        vector_store=store,
        safety=SafetyChecker(),
        history=history,
    )
    graph = RAGGraphBuilder(nodes, MemorySaver()).build()
    return graph, store, ingestion


def main() -> None:
    st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="wide")
    configure_logging()
    st.title("PDF Question Answering")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    settings = get_settings()
    if not settings.openai_api_key.get_secret_value():
        st.error("Set OPENAI_API_KEY in your .env file, then restart the app.")
        st.stop()

    graph, vector_store, ingestion = get_runtime(st.session_state.session_id)
    selected_files, clear_chat = render_sidebar(
        vector_store,
        ingestion,
        settings.upload_dir / st.session_state.session_id,
    )
    if clear_chat:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

    render_chat(graph, st.session_state, selected_files, settings.log_dir)


if __name__ == "__main__":
    main()