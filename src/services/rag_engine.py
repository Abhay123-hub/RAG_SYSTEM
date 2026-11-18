from __future__ import annotations

import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from threading import RLock
from uuid import UUID

from src.contracts.engine import RAGEngine
from src.contracts.events import CompletedEvent, QueryStreamEvent, TokenEvent
from src.contracts.schemas import (
    DocumentInfo,
    IngestResult,
    QueryResult,
    SourceCitation,
)
from src.core.context import request_context
from src.core.exceptions import IngestionError


class _SessionRuntime:
    def __init__(self, session_id: str) -> None:
        from langgraph.checkpoint.memory import MemorySaver

        from config.settings import get_settings
        from src.agents.graph_builder import RAGGraphBuilder
        from src.agents.nodes import RAGNodes
        from src.agents.router import QueryRouter
        from src.embeddings.embedder import EmbeddingService
        from src.generation.answer_generator import AnswerGenerator
        from src.generation.citation_builder import CitationBuilder
        from src.generation.query_rewriter import QueryRewriter
        from src.guardrails.safety import SafetyChecker
        from src.ingestion.chunker import DocumentChunker
        from src.ingestion.file_registry import FileRegistry
        from src.ingestion.ingestion_service import IngestionService
        from src.ingestion.pdf_loader import PDFLoader
        from src.memory.history_manager import HistoryManager
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.reranker import Reranker
        from src.retrieval.relevance_grader import RelevanceGrader
        from src.vectorstore.chroma_store import ChromaVectorStore

        settings = get_settings()
        embeddings = EmbeddingService()
        self.vector_store = ChromaVectorStore(session_id, embeddings)
        self.ingestion = IngestionService(
            PDFLoader(),
            DocumentChunker(),
            FileRegistry(),
            self.vector_store,
        )
        history = HistoryManager(settings.history_turns)
        nodes = RAGNodes(
            router=QueryRouter(),
            rewriter=QueryRewriter(history_manager=history),
            retriever=HybridRetriever(self.vector_store),
            reranker=Reranker(),
            grader=RelevanceGrader(),
            generator=AnswerGenerator(),
            citation_builder=CitationBuilder(),
            vector_store=self.vector_store,
            safety=SafetyChecker(),
            history=history,
        )
        self.graph = RAGGraphBuilder(nodes, MemorySaver()).build()


class RAGEngineService(RAGEngine):
    """Adapt the existing RAG graph and document store to the shared API contract."""

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionRuntime] = {}
        self._lock = RLock()

    def _runtime(self, session_id: str) -> _SessionRuntime:
        with self._lock:
            runtime = self._sessions.get(session_id)
            if runtime is None:
                raise KeyError(session_id)
            return runtime

    def initialize_session(self, *, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = _SessionRuntime(session_id)

    def list_documents(self, *, session_id: str) -> list[DocumentInfo]:
        runtime = self._runtime(session_id)
        return [
            DocumentInfo(
                document_id=str(item["file_hash"]),
                file_name=str(item["file_name"]),
                indexed_page_count=int(item["pages"]),
                chunk_count=int(item["chunks"]),
            )
            for item in runtime.vector_store.list_files()
        ]

    def ingest_document(
        self, *, session_id: str, file_name: str, content: bytes
    ) -> IngestResult:
        runtime = self._runtime(session_id)
        from config.settings import get_settings

        settings = get_settings()
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix="api-", dir=settings.upload_dir
            ) as temporary_directory:
                path = Path(temporary_directory) / file_name
                path.write_bytes(content)
                result = runtime.ingestion.ingest(path)
        except IngestionError as error:
            if "No selectable text found" in str(error):
                raise ValueError("The PDF contains no extractable text.") from error
            if "Could not index" in str(error):
                raise OSError("The PDF could not be indexed.") from error
            raise RuntimeError("The uploaded file is not a valid, readable PDF.") from error

        document_id = str(result["file_hash"])
        if result["duplicate"]:
            existing = next(
                (
                    document
                    for document in self.list_documents(session_id=session_id)
                    if document.document_id == document_id
                ),
                None,
            )
            if existing is None:
                raise RuntimeError("The duplicate document could not be retrieved.")
            return IngestResult(document=existing, duplicate=True)

        return IngestResult(
            document=DocumentInfo(
                document_id=document_id,
                file_name=file_name,
                indexed_page_count=int(result["pages"]),
                chunk_count=int(result["chunks"]),
            ),
            duplicate=False,
        )

    def delete_document(self, *, session_id: str, document_id: str) -> None:
        runtime = self._runtime(session_id)
        runtime.vector_store.store._collection.delete(
            where={"file_hash": document_id}
        )

    def _query(
        self,
        *,
        runtime: _SessionRuntime,
        session_id: str,
        conversation_id: UUID,
        request_id: UUID,
        query: str,
        document_ids: Sequence[str] | None,
        include_trace: bool,
        streaming: bool,
    ) -> dict[str, object]:
        filters: dict[str, object] = {}
        if document_ids is not None:
            filters = {"file_hash": {"$in": list(document_ids)}}
        with request_context(str(request_id), session_id):
            from langchain_core.messages import HumanMessage

            return runtime.graph.invoke(
                {
                    "question": query,
                    "messages": [HumanMessage(content=query)],
                    "request_id": str(request_id),
                    "streaming": streaming,
                    "filters": filters,
                },
                {"configurable": {"thread_id": str(conversation_id)}},
            )

    @staticmethod
    def _query_result(
        state: dict[str, object],
        *,
        include_trace: bool,
    ) -> QueryResult:
        route = str(state.get("route", "general_chat"))
        raw_error = state.get("error")
        status = (
            "insufficient_context"
            if raw_error == "insufficient_context"
            else "refused"
            if route == "out_of_scope"
            else "answered"
        )
        chunks = state.get("chunks", [])
        citations: list[SourceCitation] = []
        if isinstance(chunks, list):
            for source in state.get("sources", []):
                matching_chunk = next(
                    (
                        chunk
                        for chunk in chunks
                        if chunk.file_name == source.file_name
                        and chunk.page == source.page
                    ),
                    None,
                )
                if matching_chunk is None:
                    continue
                citations.append(
                    SourceCitation(
                        document_id=matching_chunk.file_hash,
                        file_name=source.file_name,
                        page=source.page,
                        snippet=source.snippet,
                        score=source.score,
                    )
                )
        return QueryResult(
            conversation_id=UUID(str(state.get("conversation_id")))
            if state.get("conversation_id")
            else UUID(int=0),
            status=status,
            route=route,
            answer=str(state.get("answer", "")),
            citations=citations,
            trace=state.get("trace") if include_trace else None,
        )

    def query(
        self,
        *,
        session_id: str,
        conversation_id: UUID,
        request_id: UUID,
        query: str,
        document_ids: Sequence[str] | None,
        include_trace: bool,
    ) -> QueryResult:
        runtime = self._runtime(session_id)
        state = self._query(
            runtime=runtime,
            session_id=session_id,
            conversation_id=conversation_id,
            request_id=request_id,
            query=query,
            document_ids=document_ids,
            include_trace=include_trace,
            streaming=False,
        )
        return self._query_result(
            {**state, "conversation_id": str(conversation_id)},
            include_trace=include_trace,
        )

    def stream_query(
        self,
        *,
        session_id: str,
        conversation_id: UUID,
        request_id: UUID,
        query: str,
        document_ids: Sequence[str] | None,
        include_trace: bool,
    ) -> Iterator[QueryStreamEvent]:
        runtime = self._runtime(session_id)
        from langchain_core.messages import HumanMessage

        filters: dict[str, object] = {}
        if document_ids is not None:
            filters = {"file_hash": {"$in": list(document_ids)}}
        state: dict[str, object] = {}
        inputs = {
            "question": query,
            "messages": [HumanMessage(content=query)],
            "request_id": str(request_id),
            "streaming": True,
            "filters": filters,
        }
        config = {"configurable": {"thread_id": str(conversation_id)}}
        with request_context(str(request_id), session_id):
            for mode, payload in runtime.graph.stream(
                inputs, config, stream_mode=["custom", "updates"]
            ):
                if mode == "custom" and isinstance(payload, dict):
                    token = payload.get("token")
                    if token:
                        yield TokenEvent(delta=str(token))
                elif mode == "updates" and isinstance(payload, dict):
                    for update in payload.values():
                        if isinstance(update, dict):
                            state.update(update)
        state["conversation_id"] = str(conversation_id)
        yield CompletedEvent(
            result=self._query_result(
                state, include_trace=include_trace
            )
        )
