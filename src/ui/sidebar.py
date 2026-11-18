import streamlit as st

from src.api_client import APIClient, APIError


def render_sidebar(client: APIClient) -> tuple[list[str], bool]:
    clear_chat = False
    with st.sidebar:
        st.header("Documents")
        uploads = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Maximum 50 MB per PDF",
        )
        if st.button("Upload PDFs", icon=":material/upload_file:", disabled=not uploads):
            progress = st.progress(0)
            for index, uploaded in enumerate(uploads or []):
                try:
                    result = client.upload_document(uploaded.name, uploaded.getvalue())
                    document = result.document
                    label = "Already indexed" if result.duplicate else (
                        f"Indexed {document.indexed_page_count} pages, "
                        f"{document.chunk_count} chunks"
                    )
                    st.success(f"{uploaded.name}: {label}")
                except APIError as error:
                    st.error(f"{uploaded.name}: {error.message}")
                except Exception as error:
                    st.error(f"{uploaded.name}: {error}")
                finally:
                    progress.progress((index + 1) / len(uploads or []))
            progress.empty()
            st.rerun()

        try:
            document_response = client.list_documents()
            documents = document_response.documents
        except APIError as error:
            st.error(f"Could not list documents: {error.message}")
            documents = []
        except Exception as error:
            st.error(f"Could not list documents: {error}")
            documents = []

        label_to_id = {document.file_name: document.document_id for document in documents}
        selected_names = st.multiselect(
            "Search in",
            list(label_to_id),
            help="Leave empty to search across all documents.",
        )
        selected = [label_to_id[name] for name in selected_names]

        for document in documents:
            with st.container():
                st.write(f"**{document.file_name}**")
                st.caption(
                    f"{document.indexed_page_count} pages · "
                    f"{document.chunk_count} chunks"
                )
                if st.button(
                    "Delete",
                    key=f"delete_{document.document_id}",
                    icon=":material/delete:",
                ):
                    try:
                        client.delete_document(document.document_id)
                        st.success(f"Deleted {document.file_name}")
                        st.rerun()
                    except APIError as error:
                        st.error(f"Could not delete {document.file_name}: {error.message}")
                    except Exception as error:
                        st.error(f"Could not delete {document.file_name}: {error}")

        st.divider()
        clear_chat = st.button("Clear chat", icon=":material/delete_sweep:")
    return selected, clear_chat