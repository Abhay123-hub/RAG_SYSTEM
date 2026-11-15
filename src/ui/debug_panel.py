import streamlit as st


def render_debug(state: dict[str, object]) -> None:
    with st.expander("Request trace"):
        trace = state.get("trace", {})
        st.json(trace if isinstance(trace, dict) else {})
        chunks = state.get("chunks", [])
        if chunks:
            st.markdown("**Retrieved passages**")
            for chunk in chunks:
                st.caption(f"{chunk.file_name}, p.{chunk.page} · score {chunk.score or 0:.3f}")
                st.write(chunk.text)