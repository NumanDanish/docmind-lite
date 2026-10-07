import streamlit as st
from rag_engine import get_client, load_chunks, embed_all, search, answer, MAX_PAGES

st.set_page_config(page_title="DocMind Lite", page_icon="📄")
st.title("📄 DocMind Lite")
st.caption("Upload a PDF and ask questions. Answers include page numbers.")

client = get_client(st.secrets["GEMINI_API_KEY"])

uploaded = st.file_uploader("Upload a PDF", type="pdf")

# Process a new PDF only once
if uploaded and st.session_state.get("file_name") != uploaded.name:
    with st.spinner("Reading the PDF..."):
        chunks, total_pages = load_chunks(uploaded)

    if not chunks:
        st.error("No text found. This may be a scanned PDF.")
        st.stop()

    bar = st.progress(0.0, text="Preparing the document...")

    def update(done, total):
        bar.progress(done / total, text=f"Processing {done}/{total} chunks")

    try:
        vectors = embed_all(client, [c["text"] for c in chunks], on_progress=update)
    except Exception as e:
        st.error(f"Something went wrong while processing: {e}")
        st.stop()

    bar.empty()
    st.session_state.file_name = uploaded.name
    st.session_state.chunks = chunks
    st.session_state.vectors = vectors
    st.session_state.messages = []

    if total_pages > MAX_PAGES:
        st.warning(f"This PDF has {total_pages} pages. Only the first {MAX_PAGES} were used.")
    st.success(f"Ready! {len(chunks)} pieces from {min(total_pages, MAX_PAGES)} pages.")

# Wait for a PDF before showing the chat
if "chunks" not in st.session_state:
    st.info("👆 Upload a PDF to get started.")
    st.stop()

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# The chat input
question = st.chat_input("Ask a question about your PDF")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document..."):
                top = search(client, question, st.session_state.chunks, st.session_state.vectors)
                reply = answer(client, question, top)
        except Exception as e:
            reply = f"Sorry, something went wrong: {e}"
            top = []

        st.markdown(reply)
        if top:
            with st.expander("📑 Sources"):
                for score, c in top:
                    st.markdown(f"**Page {c['page']}** (match {score:.2f})")
                    st.caption(c["text"][:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": reply})