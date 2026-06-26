import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_FAISS_PATH = Path("models/faiss")

def get_faiss_path(chat_id: str) -> Path:
    """Each chat gets its own FAISS folder."""
    return BASE_FAISS_PATH / chat_id

def build_vectorstore(chunks: list, chat_id: str):
    from langchain_community.vectorstores import FAISS
    from src.embeddings.embedder import get_embeddings

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    faiss_path = get_faiss_path(chat_id)
    faiss_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(faiss_path))
    print(f"✅ Vector store saved for chat {chat_id}")
    return vectorstore

def load_vectorstore(chat_id: str):
    from langchain_community.vectorstores import FAISS
    from src.embeddings.embedder import get_embeddings

    faiss_path = get_faiss_path(chat_id)
    if not (faiss_path / "index.faiss").exists():
        raise FileNotFoundError(f"No FAISS index found for chat {chat_id}. Please upload a PDF first.")

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(faiss_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

def get_retriever(chat_id: str):
    k = int(os.getenv("RETRIEVAL_K", 5))
    vectorstore = load_vectorstore(chat_id)
    return vectorstore.as_retriever(search_kwargs={"k": k})