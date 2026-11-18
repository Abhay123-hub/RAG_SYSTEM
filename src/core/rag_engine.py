import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence
from uuid import UUID

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from config.settings import get_settings
from src.agents.graph_builder import RAGGraphBuilder
from src.agents.nodes import RAGNodes
from src.agents.router import QueryRouter
from src.contracts.events import CompletedEvent, TokenEvent
from src.contracts.schemas import DocumentInfo, IngestResult, QueryResult, SourceCitation
from src.core.exceptions import DocumentNotFoundError, SessionNotFoundError
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


@dataclass
class _SessionRuntime:
    store: object
    ingestion: IngestionService
    graph: object
    checkpointer: MemorySaver = field(default_factory=MemorySaver)


class LangGraphRAGEngine:
    """Concrete RAGEngine implementation backed by the existing LangGraph pipeline."""

    def __init__(
        self,
        *,
        embeddings: object | None = None,
        vector_store_factory: object | None = None,
        graph_factory: object | None = None,
        upload_dir: Path | None = None,
        store: object | None = None,
        graph: object | None = None,
        embedding_service: object | None = None,
    ) -> None:
        self._settings = get_settings()
        self._embeddings = embeddings or embedding_service
        self._vector_store_factory = vector_store_factory
        self._graph_factory = graph_factory
        self._upload_dir = upload_dir or self._settings.upload_dir
        self._default_store = store
        self._default_graph = graph
        self._sessions: dict[str, _SessionRuntime] = {}

    def initialize_session(self, *, session_id: str) -> None:
        if not session_id:
            raise ValueError("session_id must not be empty")
        if session_id in self._sessions:
            return

        embeddings = self._embeddings or EmbeddingService()
        if self._default_store is not None:
            store = self._default_store
        elif self._vector_store_factory is not None:
            store = self._vector_store_factory(session_id, embeddings)
        else:
            store = ChromaVectorStore(session_id, embeddings)

        registry = FileRegistry()
        ingestion = IngestionService(PDFLoader(), DocumentChunker(), registry, store)
        checkpointer = MemorySaver()
        if self._default_graph is not None:
            graph = self._default_graph
        else:
            history = HistoryManager(self._settings.history_turns)
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
            if self._graph_factory is not None:
                graph = self._graph_factory(nodes, checkpointer)
            else:
                graph = RAGGraphBuilder(nodes, checkpointer).build()
        self._sessions[session_id] = _SessionRuntime(store, ingestion, graph, checkpointer)

    def _runtime(self, session_id: str) -> _SessionRuntime:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise SessionNotFoundError(f"Session '{session_id}' was not initialized.") from error

    def list_documents(self, *, session_id: str) -> list[DocumentInfo]:
        runtime = self._runtime(session_id)
        records = runtime.store.list_files()
        documents = [
            DocumentInfo(
                document_id=str(record["file_hash"]),
                file_name=str(record["file_name"]),
                indexed_page_count=int(record["pages"]),
                chunk_count=int(record["chunks"]),
            )
            for record in records
        ]
        return sorted(documents, key=lambda document: (document.file_name.casefold(), document.document_id))

    @staticmethod
    def _sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _conflict_name(file_name: str, existing_names: set[str]) -> str:
        if file_name not in existing_names:
            return file_name
        path = Path(file_name)
        stem = path.stem
        suffix = path.suffix
        index = 2
        while True:
            candidate = f"{stem} ({index}){suffix}"
            if candidate not in existing_names:
                return candidate
            index += 1

    def ingest_document(self, *, session_id: str, file_name: str, content: bytes) -> IngestResult:
        runtime = self._runtime(session_id)
        if not file_name.casefold().endswith(".pdf"):
            raise ValueError("The uploaded filename must end in .pdf.")
        if not content:
            raise ValueError("The uploaded PDF is empty.")

        document_hash = self._sha256(content)
        existing = runtime.store.list_files()
        for record in existing:
            if str(record["file_hash"]) == document_hash:
                document = DocumentInfo(
                    document_id=document_hash,
                    file_name=str(record["file_name"]),
                    indexed_page_count=int(record["pages"]),
                    chunk_count=int(record["chunks"]),
                )
                return IngestResult(document=document, duplicate=True)

        resolved_name = self._conflict_name(file_name, {str(record["file_name"]) for record in existing})
        upload_dir = self._upload_dir / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / resolved_name
        target.write_bytes(content)

        try:
            result = runtime.ingestion.ingest(target)
        except Exception:
            if target.exists():
                target.unlink()
            raise

        runtime.ingestion.registry.register(target, int(result["pages"]), int(result["chunks"]))
        document = DocumentInfo(
            document_id=document_hash,
            file_name=resolved_name,
            indexed_page_count=int(result["pages"]),
            chunk_count=int(result["chunks"]),
        )
        return IngestResult(document=document, duplicate=False)

    def delete_document(self, *, session_id: str, document_id: str) -> None:
        runtime = self._runtime(session_id)
        records = runtime.store.list_files()
        record = next((item for item in records if str(item["file_hash"]) == document_id), None)
        if record is None:
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

        try:
            if hasattr(runtime.store, "delete_by_document_id"):
                runtime.store.delete_by_document_id(document_id)
            else:
                runtime.store.delete_by_file(str(record["file_name"]))
        except NotImplementedError:
            runtime.store.delete_by_file(str(record["file_name"]))
        runtime.ingestion.registry.remove(document_id)

        source = self._upload_dir / session_id / str(record["file_name"])
        if source.exists():
            source.unlink()

    def _filters(self, runtime: _SessionRuntime, document_ids: Sequence[str] | None) -> dict[str, object]:
        if document_ids is None:
            return {}
        ids = list(document_ids)
        known = {str(record["file_hash"]) for record in runtime.store.list_files()}
        missing = [document_id for document_id in ids if document_id not in known]
        if missing:
            raise DocumentNotFoundError(f"Document '{missing[0]}' was not found.")
        return {"file_hash": {"$in": ids}}

    def _source_citations(self, runtime: _SessionRuntime, sources: list[object]) -> list[SourceCitation]:
        chunks = runtime.store.all_chunks()
        hash_by_name_page = {
            (chunk.file_name.casefold(), chunk.page): chunk.file_hash
            for chunk in chunks
        }
        citations: list[SourceCitation] = []
        for source in sources:
            file_name = str(getattr(source, "file_name", ""))
            page = int(getattr(source, "page", 1))
            snippet = str(getattr(source, "snippet", ""))
            score = getattr(source, "score", None)
            document_id = hash_by_name_page.get((file_name.casefold(), page))
            if document_id is None:
                continue
            citations.append(
                SourceCitation(
                    document_id=document_id,
                    file_name=file_name,
                    page=page,
                    snippet=snippet,
                    score=float(score) if score is not None else None,
                )
            )
        return citations

    def _execute(
        self,
        *,
        session_id: str,
        conversation_id: UUID,
        request_id: UUID,
        query: str,
        document_ids: Sequence[str] | None,
        include_trace: bool,
        streaming: bool,
    ) -> tuple[dict[str, object], list[object]]:
        runtime = self._runtime(session_id)
        filters = self._filters(runtime, document_ids)
        initial = {
            "question": query,
            "messages": [HumanMessage(content=query)],
            "streaming": streaming,
            "filters": filters,
            "request_id": str(request_id),
        }
        config = {"configurable": {"thread_id": str(conversation_id)}}
        result = runtime.graph.invoke(initial, config)
        sources = list(result.get("sources", []))
        return result, sources

    def _make_result(
        self,
        *,
        conversation_id: UUID,
        result: dict[str, object],
        runtime: _SessionRuntime,
        sources: list[object],
        include_trace: bool,
    ) -> QueryResult:
        route = str(result.get("route", "rag_query"))
        if route == "out_of_scope":
            status = "refused"
        elif route == "general_chat":
            status = "answered"
        elif result.get("error") == "insufficient_context" or not sources:
            status = "insufficient_context"
        else:
            status = "answered"

        return QueryResult(
            conversation_id=conversation_id,
            status=status,
            route=route,
            answer=str(result.get("answer", "")),
            citations=self._source_citations(runtime, sources),
            trace=dict(result.get("trace", {})) if include_trace else None,
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
        result, sources = self._execute(
            session_id=session_id,
            conversation_id=conversation_id,
            request_id=request_id,
            query=query,
            document_ids=document_ids,
            include_trace=include_trace,
            streaming=False,
        )
        return self._make_result(
            conversation_id=conversation_id,
            result=result,
            runtime=runtime,
            sources=sources,
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
    ) -> Iterator[TokenEvent | CompletedEvent]:
        runtime = self._runtime(session_id)
        filters = self._filters(runtime, document_ids)
        initial = {
            "question": query,
            "messages": [HumanMessage(content=query)],
            "streaming": True,
            "filters": filters,
            "request_id": str(request_id),
        }
        config = {"configurable": {"thread_id": str(conversation_id)}}

        state: dict[str, object] = {}
        try:
            stream = runtime.graph.stream(initial, config, stream_mode=["updates", "custom"])
            for event in stream:
                mode = event[0] if isinstance(event, tuple) else "updates"
                payload = event[1] if isinstance(event, tuple) else event
                if mode == "custom":
                    if isinstance(payload, dict) and "token" in payload:
                        yield TokenEvent(delta=str(payload["token"]))
                    continue
                if isinstance(payload, dict):
                    for key, value in payload.items():
                        if isinstance(value, dict):
                            state.update(value)
                        else:
                            state[key] = value
        except TypeError:
            stream = runtime.graph.stream(initial, config)
            for event in stream:
                if isinstance(event, dict):
                    if "token" in event:
                        yield TokenEvent(delta=str(event["token"]))
                    else:
                        state.update(event)
        except NotImplementedError:
            result, sources = self._execute(
                session_id=session_id,
                conversation_id=conversation_id,
                request_id=request_id,
                query=query,
                document_ids=document_ids,
                include_trace=include_trace,
                streaming=True,
            )
            state = result

        if not state:
            result, sources = self._execute(
                session_id=session_id,
                conversation_id=conversation_id,
                request_id=request_id,
                query=query,
                document_ids=document_ids,
                include_trace=include_trace,
                streaming=False,
            )
            state = result

        if "sources" not in state:
            result, sources = self._execute(
                session_id=session_id,
                conversation_id=conversation_id,
                request_id=request_id,
                query=query,
                document_ids=document_ids,
                include_trace=include_trace,
                streaming=False,
            )
            state = result
            sources = list(result.get("sources", []))
        else:
            sources = list(state.get("sources", []))

        completed = self._make_result(
            conversation_id=conversation_id,
            result=state,
            runtime=runtime,
            sources=sources,
            include_trace=include_trace,
        )
        yield CompletedEvent(result=completed)


RAGEngine = LangGraphRAGEngine
RAGEngineImpl = LangGraphRAGEngine
LangGraphEngine = LangGraphRAGEngine
