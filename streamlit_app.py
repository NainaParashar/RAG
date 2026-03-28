from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from src.pipeline import QAPipeline


def persist_uploaded_pdf(uploaded_file) -> str:
    """Save uploaded PDF to a local folder and return its absolute path."""
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Keep filename simple/safe for local filesystem use.
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", uploaded_file.name)
    output_path = uploads_dir / safe_name
    output_path.write_bytes(uploaded_file.getbuffer())
    return str(output_path.resolve())


def render_source(source: dict, source_pdf_path: str | None) -> None:
    """Render one source row with clickable page link when possible."""
    page = source["page_start"]
    section = source["section_number"]
    title = source["section_title"]
    chunk_type = source["chunk_type"]
    preview = source["preview"]

    st.markdown(f"**Section {section}** ({title}) | Page `{page}` | Type `{chunk_type}`")

    # If we know the local source PDF path, create a file URL with a page anchor.
    # This is convenient during demo because evaluators can verify source quickly.
    if source_pdf_path and Path(source_pdf_path).exists():
        file_url = f"file://{Path(source_pdf_path).resolve()}#page={page}"
        st.markdown(f"[Open source page]({file_url})")

    with st.expander("Preview text"):
        st.write(preview)


def main() -> None:
    st.set_page_config(page_title="Technical Manual QA", layout="wide")
    st.title("Problem 2 - Technical Manual QA")
    st.caption("Smart ingestion + hybrid retrieval + grounded synthesis")

    # Use session state so app remembers selected paths between reruns.
    if "source_pdf_path" not in st.session_state:
        st.session_state.source_pdf_path = ""
    if "light_mode" not in st.session_state:
        st.session_state.light_mode = True
    if "pipe" not in st.session_state:
        st.session_state.pipe = QAPipeline(light_mode=st.session_state.light_mode)

    with st.sidebar:
        st.subheader("Knowledge Base")
        selected_mode = st.checkbox("Light mode (faster startup)", value=st.session_state.light_mode)
        if selected_mode != st.session_state.light_mode:
            st.session_state.light_mode = selected_mode
            # Recreate pipeline to apply mode switch.
            st.session_state.pipe = QAPipeline(light_mode=st.session_state.light_mode)

        auto_build = st.checkbox("Auto-build after upload", value=False)
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_pdf is not None:
            try:
                saved_path = persist_uploaded_pdf(uploaded_pdf)
                st.session_state.source_pdf_path = saved_path
                st.success(f"Uploaded: {Path(saved_path).name}")
                if auto_build:
                    stats = st.session_state.pipe.build(
                        st.session_state.source_pdf_path,
                    )
                    st.success(
                        f"KB auto-built: {stats['chunks']} chunks, "
                        f"{stats['acronyms']} acronyms ({'light' if stats['light_mode'] else 'full'} mode)"
                    )
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

        source_pdf_path = st.text_input(
            "PDF path",
            value=st.session_state.source_pdf_path,
            placeholder="/Users/you/Downloads/nasa_systems_engineering_handbook_0.pdf",
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Build KB", use_container_width=True):
                try:
                    stats = st.session_state.pipe.build(source_pdf_path)
                    st.session_state.source_pdf_path = source_pdf_path
                    st.success(
                        f"KB built: {stats['chunks']} chunks, {stats['acronyms']} acronyms "
                        f"({'light' if stats['light_mode'] else 'full'} mode)"
                    )
                except Exception as exc:
                    st.error(f"Build failed: {exc}")
        with col2:
            if st.button("Load Artifacts", use_container_width=True):
                try:
                    stats = st.session_state.pipe.load()
                    st.success(
                        f"Loaded artifacts: {stats['chunks']} chunks, {stats['acronyms']} acronyms "
                        f"({'light' if stats['light_mode'] else 'full'} mode)"
                    )
                except Exception as exc:
                    st.error(f"Load failed: {exc}")

    arch_path = Path("/Users/nainaparashar/Downloads/nasa_qa_system_architecture.svg")
    if arch_path.exists():
        st.image(str(arch_path), caption="System architecture")

    st.subheader("Ask a question")
    question = st.text_input(
        "Question",
        placeholder="How does risk management feed into technical reviews?",
    )

    top_k = st.slider("Top-k chunks", min_value=4, max_value=16, value=8, step=1)

    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        try:
            result = st.session_state.pipe.ask(question, top_k=top_k)

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources")
            for source in result["sources"][:6]:
                render_source(source, st.session_state.source_pdf_path)
                st.divider()

        except Exception as exc:
            st.error(f"Could not answer question: {exc}")


if __name__ == "__main__":
    main()
