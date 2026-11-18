from typing import Any

import streamlit as st

from src.api_client import APIClient, APIError
from src.contracts.schemas import QueryResponse, SourceCitation
from src.ui.debug_panel import render_debug


def _show_citations(citations: list[SourceCitation]) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    for citation in citations:
        label = f"{citation.file_name}, p. {citation.page}"
        if citation.score is not None:
            label += f" · score {citation.score:.3f}"
        with st.expander(label):
            st.write(citation.snippet)


def render_chat(client: APIClient, session_state: Any, selected_documents: list[str]) -> None:
    for turn in session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            for citation in turn.get("citations", []):
                if isinstance(citation, dict):
                    try:
                        citation = SourceCitation.model_validate(citation)
                    except Exception:
                        continue
                if isinstance(citation, SourceCitation):
                    with st.expander(f"{citation.file_name}, p. {citation.page}"):
                        st.write(citation.snippet)

    question = st.chat_input("Ask a question about your PDFs")
    if not question:
        return

    session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            citations: list[SourceCitation] = []
            trace: dict[str, object] | None = None
            answer_parts: list[str] = []
            placeholder = st.empty()
            streaming_error: APIError | None = None

            try:
                with st.spinner("Generating answer..."):
                    stream = client.stream_query(
                        query=question,
                        conversation_id=session_state.conversation_id,
                        document_ids=selected_documents or None,
                        include_trace=True,
                    )
                    for event in stream:
                        name = str(event.get("event", ""))
                        data = event.get("data", {})
                        if not isinstance(data, dict):
                            continue
                        if name in {"token", "message"}:
                            delta = data.get("delta")
                            if isinstance(delta, str):
                                answer_parts.append(delta)
                                placeholder.markdown("".join(answer_parts))
                            if "result" in data and isinstance(data["result"], dict):
                                name = "done"
                                data = data["result"]
                            elif "answer" in data:
                                name = "done"
                        if name == "citations":
                            citations = [
                                SourceCitation.model_validate(item)
                                for item in data.get("citations", [])
                                if isinstance(item, dict)
                            ]
                        elif name in {"done", "completed"}:
                            done_data = data
                            if "result" in data and isinstance(data["result"], dict):
                                done_data = data["result"]
                            try:
                                result = QueryResponse.model_validate(done_data)
                                answer_parts = [result.answer]
                                citations = result.citations
                                trace = result.trace
                                placeholder.markdown(result.answer)
                                session_state.conversation_id = str(result.conversation_id)
                            except Exception:
                                answer = data.get("answer")
                                if isinstance(answer, str):
                                    answer_parts = [answer]
                                    placeholder.markdown(answer)
                                conversation_id = data.get("conversation_id")
                                if isinstance(conversation_id, str):
                                    session_state.conversation_id = conversation_id
                                citations_data = data.get("citations")
                                if isinstance(citations_data, list):
                                    citations = [
                                        SourceCitation.model_validate(item)
                                        for item in citations_data
                                        if isinstance(item, dict)
                                    ]
                                trace = data.get("trace") if isinstance(data.get("trace"), dict) else None
            except APIError as error:
                streaming_error = error
            except Exception as error:
                streaming_error = APIError(str(error))

            if streaming_error is not None:
                try:
                    with st.spinner("Generating answer..."):
                        result = client.query(
                            query=question,
                            conversation_id=session_state.conversation_id,
                            document_ids=selected_documents or None,
                            include_trace=True,
                        )
                    answer_parts = [result.answer]
                    citations = result.citations
                    trace = result.trace
                    session_state.conversation_id = str(result.conversation_id)
                    placeholder.markdown(result.answer)
                    streaming_error = None
                except APIError:
                    pass
                except Exception:
                    pass

            if streaming_error is not None:
                st.error(streaming_error.message)
                answer = "I couldn't complete that request. Please try again."
                placeholder.markdown(answer)
                session_state.chat_history.append({"role": "assistant", "content": answer})
                return

            answer = "".join(answer_parts).strip()
            if not answer:
                answer = "The request could not be completed. Please try again."
                placeholder.markdown(answer)

            _show_citations(citations)
            if trace:
                render_debug({"trace": trace})

            session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": [citation.model_dump() for citation in citations],
                }
            )
        except APIError as error:
            st.error(error.message)
        except Exception as error:
            st.error(f"I couldn't complete that request: {error}")