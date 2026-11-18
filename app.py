import uuid

import streamlit as st

from src.api_client import APIClient, APIError
from src.ui.chat_view import render_chat
from src.ui.sidebar import render_sidebar


BACKEND_URL = "http://127.0.0.1:8000"


def main() -> None:
    st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="wide")
    st.title("PDF Question Answering")

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient(BACKEND_URL)
    client: APIClient = st.session_state.api_client

    try:
        client.ensure_session()
    except APIError as error:
        st.error(f"Cannot connect to the backend: {error.message}")
        st.caption(f"Backend: {BACKEND_URL}")
        st.stop()
    except Exception as error:
        st.error(f"Cannot connect to the backend: {error}")
        st.stop()

    selected_documents, clear_chat = render_sidebar(client)
    if clear_chat:
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

    render_chat(client, st.session_state, selected_documents)


if __name__ == "__main__":
    main()