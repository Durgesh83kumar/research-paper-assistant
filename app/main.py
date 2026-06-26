import sys
import os
import uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tempfile
import streamlit as st
from dotenv import load_dotenv

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import split_documents
from src.vectorstore.faiss_store import build_vectorstore
from src.chains.summarization import summarize_paper
from src.chains.rag_chain import answer_question
from src.storage.chat_store import save_chats, load_chats, delete_chat
from src.evaluation.evaluator import evaluate_answer, evaluate_general_answer, render_scores

load_dotenv()

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# ---- Load chats from disk on first run ----
if "chats" not in st.session_state:
    chats, saved_chat_id = load_chats()
    st.session_state.chats = chats
    st.session_state.current_chat_id = saved_chat_id

def new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "pdf_processed": False,
        "pdf_name": None,
    }
    st.session_state.current_chat_id = chat_id
    save_chats(st.session_state.chats, chat_id)

# Create first chat if none exists
if not st.session_state.chats:
    new_chat()

# Set current chat if not set
if st.session_state.current_chat_id not in st.session_state.chats:
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[-1]

# ---- Sidebar ----
with st.sidebar:
    st.title("🔬 Research Assistant")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_chat()
        st.rerun()

    st.markdown("---")
    st.markdown("**Recent Chats**")

    for chat_id in reversed(list(st.session_state.chats.keys())):
        chat = st.session_state.chats[chat_id]
        is_active = chat_id == st.session_state.current_chat_id

        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            if st.button(
                chat["title"],
                key=f"chat_{chat_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.current_chat_id = chat_id
                save_chats(st.session_state.chats, chat_id)  # ← save which chat is active
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{chat_id}"):
                st.session_state.chats = delete_chat(chat_id, st.session_state.chats)
                save_chats(st.session_state.chats, st.session_state.current_chat_id)
                if not st.session_state.chats:
                    new_chat()
                else:
                    st.session_state.current_chat_id = list(st.session_state.chats.keys())[-1]
                st.rerun()

# ---- Current chat shortcut ----
current_chat = st.session_state.chats[st.session_state.current_chat_id]

st.title("🔬 AI Research Assistant")

# ---- Welcome message if empty ----
if not current_chat["messages"]:
    st.info("👋 Upload a research paper PDF using the **+** button below to get a summary, "
            "or just start chatting!")

# ---- Render chat history ----
for msg in current_chat["messages"]:
    if msg["role"] == "user":
        col1, col2 = st.columns([1, 2])
        with col2:
            st.markdown(
                f'<div style="background-color:#2b5797; color:white; padding:10px 15px; '
                f'border-radius:15px; text-align:right; margin:5px 0; word-wrap:break-word;">'
                f'{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            if msg.get("scores"):
                st.markdown(render_scores(msg["scores"]), unsafe_allow_html=True)

# ---- Chat input with file upload ----
prompt = st.chat_input(
    "Message AI Research Assistant...",
    accept_file=True,
    file_type=["pdf"],
)

if prompt:
    # ---- Handle file upload ----
    if prompt.files:
        uploaded_file = prompt.files[0]

        col1, col2 = st.columns([1, 2])
        with col2:
            st.markdown(
                f'<div style="background-color:#2b5797; color:white; padding:10px 15px; '
                f'border-radius:15px; text-align:right; margin:5px 0;">'
                f'📎 Uploaded: <b>{uploaded_file.name}</b></div>',
                unsafe_allow_html=True,
            )
        current_chat["messages"].append({
            "role": "user",
            "content": f"📎 Uploaded: **{uploaded_file.name}**",
        })

        with st.chat_message("assistant"):
            with st.spinner("Reading and indexing your PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                docs = load_pdf(tmp_path)
                chunks = split_documents(docs)
                build_vectorstore(chunks, st.session_state.current_chat_id)
                os.unlink(tmp_path)

                current_chat["pdf_processed"] = True
                current_chat["pdf_name"] = uploaded_file.name

                if current_chat["title"] == "New Chat":
                    current_chat["title"] = uploaded_file.name[:30]

            with st.spinner("Generating summary..."):
                summary = summarize_paper(docs)

            response = (
                f"✅ I've read **{uploaded_file.name}** ({len(docs)} pages, "
                f"{len(chunks)} chunks indexed).\n\n"
                f"## 📋 Summary\n\n{summary}\n\n---\n"
                f"*You can now ask me anything about this paper!*"
            )
            st.markdown(response)

        current_chat["messages"].append({
            "role": "assistant",
            "content": response,
        })
        save_chats(st.session_state.chats, st.session_state.current_chat_id)

    # ---- Handle text question ----
    if prompt.text:
        col1, col2 = st.columns([1, 2])
        with col2:
            st.markdown(
                f'<div style="background-color:#2b5797; color:white; padding:10px 15px; '
                f'border-radius:15px; text-align:right; margin:5px 0;">'
                f'{prompt.text}</div>',
                unsafe_allow_html=True,
            )
        current_chat["messages"].append({
            "role": "user",
            "content": prompt.text,
        })

        if current_chat["title"] == "New Chat":
            current_chat["title"] = prompt.text[:30]

        scores = None

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer_question(
                    prompt.text,
                    current_chat["messages"][:-1],
                    current_chat["pdf_processed"],
                    st.session_state.current_chat_id,
                )

            st.markdown(result["answer"])

            # PDF Q&A evaluation
            if current_chat["pdf_processed"] and result["sources"]:
                with st.spinner("Evaluating answer quality..."):
                    context = "\n\n".join([doc.page_content for doc in result["sources"]])
                    scores = evaluate_answer(prompt.text, result["answer"], context)
                    scores["type"] = "pdf"
                    st.markdown(render_scores(scores), unsafe_allow_html=True)

                with st.expander("📎 View Source Chunks"):
                    for doc in result["sources"]:
                        pg = doc.metadata.get("page", "?")
                        st.markdown(f"**Page {int(pg) + 1}:** {doc.page_content[:300]}...")

            # General chat evaluation (skip for greetings/very short messages)
            else:
                if len(prompt.text.strip()) > 15:
                    with st.spinner("Evaluating answer quality..."):
                        scores = evaluate_general_answer(prompt.text, result["answer"])
                        st.markdown(render_scores(scores), unsafe_allow_html=True)

        current_chat["messages"].append({
            "role": "assistant",
            "content": result["answer"],
            "scores": scores,
        })
        save_chats(st.session_state.chats, st.session_state.current_chat_id)

    st.rerun()