from pathlib import Path

import streamlit as st

from src.ingestion.ingestion_service import IngestionService
from src.vectorstore.base import VectorStore


def render_sidebar(vector_store: VectorStore, ingestion: IngestionService, upload_dir: Path) -> tuple[list[str], bool]:
    clear_chat = False
    with st.sidebar:
        st.header("Documents")
        uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Process PDFs", icon=":material/upload_file:", disabled=not uploads):
            upload_dir.mkdir(parents=True, exist_ok=True)
            batch = uploads or []
            progress = st.progress(0)
            for index, uploaded in enumerate(batch):
                path = upload_dir / Path(uploaded.name).name
                try:
                    path.write_bytes(uploaded.getvalue())
                    result = ingestion.ingest(path)
                    label = "Already indexed" if result["duplicate"] else f"Indexed {result['pages']} pages, {result['chunks']} chunks"
                    if result.get("scanned"):
                        label += "; scanned pages skipped (OCR not configured)"
                    st.success(f"{uploaded.name}: {label}")
                except Exception as error:
                    st.error(f"{uploaded.name}: {error}")
                finally:
                    progress.progress((index + 1) / len(batch))
            progress.empty()

        files = vector_store.list_files()
        selected = st.multiselect("Search in", [str(file["file_name"]) for file in files])
        for file in files:
            name = str(file["file_name"])
            with st.container():
                st.write(f"**{name}**")
                st.caption(f"{file['pages']} pages · {file['chunks']} chunks")
                if st.button("Delete", key=f"delete_{file['file_hash']}", icon=":material/delete:"):
                    vector_store.delete_by_file(name)
                    st.rerun()

        st.divider()
        clear_chat = st.button("Clear chat", icon=":material/delete_sweep:")
    return selected, clear_chat