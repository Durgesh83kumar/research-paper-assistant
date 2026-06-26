from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path

def load_pdf(file_path: str) -> list:
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = Path(file_path).name
    return documents