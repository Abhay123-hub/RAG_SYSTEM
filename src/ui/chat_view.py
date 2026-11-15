import json
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage

from src.core.context import request_context
from src.core.logger import get_logger
from src.ui.debug_panel import render_debug

logger = get_logger(__name__)


def _trace_record(question: str, state: dict[str, Any], request_id: str) -> dict[str, Any]:
    chunks = state.get("chunks", [])
    return {
        "request_id": request_id,
        "question": question,
        "route": state.get("route"),
        "route_confidence": state.get("route_confidence"),
        "route_reason": state.get("route_reason"),
        "standalone_question": state.get("standalone_question"),
        "error": state.get("error"),
        "answer": state.get("answer"),
        "chunks": [
            {"file_name": chunk.file_name, "page": chunk.page, "chunk_id": chunk.chunk_id, "score": chunk.score}
            for chunk in chunks
        ],
        "sources": [source.model_dump() for source in state.get("sources", [])],
        "trace": state.get("trace", {}),
    }


def render_chat(graph: Any, session_state: Any, selected_files: list[str], log_dir: Path) -> None:
    for turn in session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("Ask a question about your PDFs")
    if not question:
        return

    session_id = session_state.session_id
    request_id = str(uuid.uuid4())
    session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    state_update: dict[str, Any] = {}
    request_config = {"configurable": {"thread_id": session_state.thread_id}}
    inputs = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "request_id": request_id,
        "streaming": True,
        "filters": {"file_name": {"$in": selected_files}} if selected_files else {},
    }

    def token_stream():
        with request_context(request_id, session_id):
            for mode, payload in graph.stream(inputs, request_config, stream_mode=["custom", "updates"]):
                if mode == "custom" and isinstance(payload, dict) and payload.get("token"):
                    yield str(payload["token"])
                elif mode == "updates" and isinstance(payload, dict):
                    for update in payload.values():
                        if isinstance(update, dict):
                            state_update.update(update)

    with st.chat_message("assistant"):
        try:
            with request_context(request_id, session_id):
                answer = st.write_stream(token_stream())
            answer = str(answer or state_update.get("answer", ""))
            if not answer:
                answer = "The request could not be completed. Please try again."
                st.markdown(answer)
            session_state.chat_history.append({"role": "assistant", "content": answer})

            for source in state_update.get("sources", []):
                with st.expander(f"{source.file_name}, p.{source.page}"):
                    st.write(source.snippet)
            render_debug(state_update)

            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "traces.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(_trace_record(question, state_update, request_id), default=str) + "\n")
        except Exception:
            logger.exception("request_failed", extra={"event": "request_failed", "data": {"request_id": request_id}})
            st.error("I couldn't complete that request. Please check the logs and try again.")
            session_state.chat_history.append({"role": "assistant", "content": "I couldn't complete that request. Please try again."})