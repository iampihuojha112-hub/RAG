# Streamlit UI: upload panel, chat window, citations.
# Filled in: Module 10
import streamlit as st

from api_client import (
    ApiError,
    ask_question,
    check_health,
    delete_document,
    list_documents,
    upload_file,
)

st.set_page_config(page_title="Chat with your documents", page_icon="📄", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "flash" not in st.session_state:
    st.session_state.flash = None


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            page = f", page {s['page_number']}" if s["page_number"] > 0 else ""
            st.markdown(f"**[{s['number']}]** {s['filename']}{page}")


st.title("📄 Chat with your documents")

try:
    check_health()
except ApiError as error:
    st.error(str(error))
    st.info("Start the backend first, then refresh this page.")
    st.stop()

with st.sidebar:
    st.header("Your documents")

    if st.session_state.flash:
        kind, text = st.session_state.flash
        getattr(st, kind)(text)
        st.session_state.flash = None

    uploaded = st.file_uploader(
        "Upload a PDF or DOCX",
        type=["pdf", "docx"],
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if st.button("Add to knowledge base", disabled=uploaded is None):
        with st.spinner("Reading and indexing the document..."):
            try:
                result = upload_file(
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "application/octet-stream",
                )
            except ApiError as error:
                st.error(str(error))
            else:
                if result["status"] == "duplicate":
                    st.session_state.flash = (
                        "warning",
                        f"{result['filename']} is already stored. Nothing added.",
                    )
                else:
                    st.session_state.flash = (
                        "success",
                        f"Added {result['filename']} ({result['chunks']} chunks).",
                    )
                st.session_state.uploader_key += 1
                st.rerun()

    st.divider()

    try:
        documents = list_documents()
    except ApiError as error:
        st.error(str(error))
        documents = []

    if not documents:
        st.info("No documents yet. Upload one to begin.")

    for doc in documents:
        left, right = st.columns([4, 1])
        left.markdown(f"**{doc['filename']}**  \n{doc['chunk_count']} chunks")
        if right.button("🗑", key=f"del_{doc['document_id']}", help="Delete this document"):
            try:
                delete_document(doc["document_id"])
                st.session_state.flash = ("success", f"Deleted {doc['filename']}.")
            except ApiError as error:
                st.session_state.flash = ("error", str(error))
            st.rerun()

    names = {d["document_id"]: d["filename"] for d in documents}
    scope = st.selectbox(
        "Search in",
        options=[None, *names],
        format_func=lambda i: "All documents" if i is None else names[i],
    )

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

question = st.chat_input("Ask a question about your documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask_question(question, document_id=scope)
            except ApiError as error:
                answer, sources = f"⚠️ {error}", []
            else:
                answer, sources = result["answer"], result["sources"]
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )