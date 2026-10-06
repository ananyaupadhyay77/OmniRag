import streamlit as st
import requests

# FastAPI Backend URL
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="OmniRAG System",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ OmniRAG: Multi-Source Production System")
st.caption("MPS-Accelerated Hybrid Search (BM25 + Dense) with Cross-Encoder Re-ranking")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: PDF Ingestion ---
with st.sidebar:
    st.header("📄 Knowledge Base Ingestion")
    st.markdown("Upload a PDF document to chunk, embed, and index into Qdrant.")
    
    uploaded_file = st.file_uploader("Select PDF File", type=["pdf"])
    
    if st.button("Upload & Index", type="primary", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Processing PDF, chunking, and indexing vectors..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_BASE_URL}/ingest", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Success! Ingested **{data['filename']}**")
                        st.json({
                            "Document ID": data["document_id"],
                            "Parent Chunks": data["parent_chunks_created"],
                            "Child Chunks": data["child_chunks_created"]
                        })
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
        else:
            st.warning("Please select a PDF file first.")

    st.divider()
    top_k = st.slider("Retrieval Top K Results", min_value=1, max_value=5, value=3)

# --- MAIN CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📚 Source Citations"):
                for cit in message["citations"]:
                    st.markdown(f"**[{cit['source_type'].upper()}] {cit['source_title']}**")
                    st.caption(f"*Excerpt:* {cit['excerpt']}")
                    st.caption(f"*Parent ID:* `{cit['parent_id']}`")
                    st.divider()

# User Query Input
if query := st.chat_input("Ask a question grounded in your documents..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Searching vectors, re-ranking, and generating response..."):
            try:
                payload = {"query": query, "top_k": top_k}
                res = requests.post(f"{API_BASE_URL}/query", json=payload)

                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    citations = data["citations"]
                    latency = data["retrieval_latency_ms"]

                    st.markdown(answer)
                    st.caption(f"⚡ Retrieval & Re-ranking Latency: `{latency:.2f} ms`")

                    if citations:
                        with st.expander("📚 Source Citations"):
                            for cit in citations:
                                st.markdown(f"**[{cit['source_type'].upper()}] {cit['source_title']}**")
                                st.caption(f"*Excerpt:* {cit['excerpt']}")
                                st.caption(f"*Parent ID:* `{cit['parent_id']}`")
                                st.divider()

                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    })
                else:
                    st.error(f"Backend returned status {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to FastAPI server at {API_BASE_URL}: {e}")